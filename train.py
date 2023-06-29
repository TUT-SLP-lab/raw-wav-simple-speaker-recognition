import os
import shutil

import hydra
import numpy as np
import torch
import wandb
from lightning import Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf
from torch import nn, optim
from torch.utils.data import DataLoader

from hansyo_ssr.data.datamodule import AudioTextDataModule
from hansyo_ssr.data.dataset import SequenceDataset
from hansyo_ssr.model.simpleSR import simpleSR


@hydra.main(config_path="conf/simpleSR", config_name="ssr", version_base="1.3")
def main(config: DictConfig):
    wandb.init()

    # TODO: Hydraへの対応
    # TODO: Pytorchへの移行

    # 各種学習データのパス
    # 学習データ等の特徴量(feats.scp)が存在するディレクトリ
    feat_dir_train = "data/fbank/train"
    feat_dir_dev = "data/fbank/dev"
    feat_dir_eval = "data/fbank/eval"
    # # 学習データ等の特徴量(wav.scp)が存在するディレクトリ
    # feat_dir_train = 'data/wav/train'
    # feat_dir_dev = 'data/wav/dev'
    # feat_dir_eval = 'data/wav/eval'

    # 実験ディレクトリ
    # train_set_name = 'train_small' or 'train_large'
    train_set_name = os.path.basename(feat_dir_train)
    exp_dir = "./exp_" + os.path.basename(feat_dir_train)

    # 学習/開発データの特徴量リストファイル
    feat_scp_train = os.path.join(feat_dir_train, "feats.scp")
    feat_scp_dev = os.path.join(feat_dir_dev, "feats.scp")
    feat_scp_eval = os.path.join(feat_dir_eval, "feats.scp")

    # 学習/開発データのラベルファイル
    label_train = "data/train/label"
    label_dev = "data/dev/label"
    label_eval = "data/eval/label"

    # トークンリスト
    token_list_path = os.path.join(exp_dir, "data", "token_list")
    # 学習結果を出力するディレクトリ
    output_dir = os.path.join(exp_dir, "simple-SR")

    # 学習パラメータ
    # batch_size = 90
    batch_size = 80
    max_num_epoch = 50
    # max_num_epoch = 5  # LINK:EPOCH_TEST_MOVE

    # 入力データのパラメータ
    feats_dim = 80  # MFCCのbin数

    # モデルパラメータ
    input_length = 128
    input_channel = 80
    conv_nums = 3  # CNN層の数
    # hidden_dim = 4096  # 隠れ層の次元数
    fc1_params = 2048
    fc2_params = 512
    initial_learning_rate = 0.05  # 開始時の学習率
    lr_decay_start_epoch = 10  # 学習率の玄瑞やESの判定を始めるエポック数
    lr_decay_factor = 0.5  # 学習率の減衰率
    early_stop_threshold = 3  # ESするのに最低損失が何epoch変化しないかを判断する

    # train, dev, evalのデータセットを作る
    datasets = []
    feat_means = []
    feat_stds = []
    for mean_path, label_path, feat_path in zip(
        [feat_dir_train, feat_dir_dev, feat_dir_eval],
        [label_train, label_dev, label_eval],
        [feat_scp_train, feat_scp_dev, feat_scp_eval],
    ):
        # 訓練データから計算された特徴量の平均/標準偏差ファイル
        mean_std_file = os.path.join(mean_path, "mean_std.txt")

        # 特徴量を読み込む
        # 特徴量の平均/標準偏差ファイルを読み込む
        with open(mean_std_file, mode="r") as f:
            # 全行読み込み
            lines = f.readlines()
            # 1行目(0始まり)が平均値ベクトル(mean)，
            # 3行目が標準偏差ベクトル(std)
            mean_line = lines[1]
            std_line = lines[3]
            # スペース区切りのリストに変換
            feat_mean = mean_line.split()
            feat_std = std_line.split()
            # numpy arrayに変換
            feat_means.append(np.array(feat_mean, dtype=np.float32))
            feat_stds.append(np.array(feat_std, dtype=np.float32))

        # 平均/標準偏差ファイルをコピーする
        os.makedirs(os.path.join(output_dir, mean_path), exist_ok=True)
        shutil.copyfile(mean_std_file, os.path.join(output_dir, mean_path, "mean_std.txt"))

        # 訓練/開発データのデータセットを作成する
        datasets.append(SequenceDataset(feat_path, label_path, feat_means[-1], feat_stds[-1], input_length))

    # 次元数の情報を得る
    feat_dim = np.size(feat_means[0])

    # NNを構築する
    dim_out = 3
    model = simpleSR(feats_dim, input_length, input_channel, conv_nums, fc1_params, fc2_params, dim_out)
    wandb.watch(model)

    # オプティマイザを定義
    optimizer = optim.Adadelta(model.parameters(), lr=initial_learning_rate, rho=0.95, eps=1e-8, weight_decay=0.0)

    # 各種データローダを作成
    train_loader = DataLoader(datasets[0], batch_size=batch_size, shuffle=True, num_workers=8)
    dev_loader = DataLoader(datasets[1], batch_size=batch_size, shuffle=False, num_workers=8)
    eval_loader = DataLoader(datasets[2], batch_size=batch_size, shuffle=False, num_workers=8)

    # クロスエントロピーを損失関数として用いる
    criterion = nn.CrossEntropyLoss()

    # CUDAが使える場合はモデルパラメータをGPUに置く
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    # modelを学習モードに設定
    model.train()

    # 訓練データと開発データをまとめとく
    dataset_loader = {"train": train_loader, "valid": dev_loader}

    # 性能が良いやつを保持しておくための変数
    best_loss = -1
    best_model = None
    best_epoch = 0

    # 学習時のlogを残しとくためのディレクトリを作る
    log_dir = os.path.join(output_dir, "log")
    os.makedirs(log_dir, exist_ok=True)

    early_stop_flag = False
    for epoch in range(max_num_epoch):
        print(f"epoch_num: {epoch}")
        if early_stop_flag:  # ESする
            break

        for phase in ["train", "valid"]:
            total_loss = 0
            total_utt = 0
            total_error = 0
            total_frames = 0

            for features, labels, feat_len, label_len, utt_ids in dataset_loader[phase]:
                features, labels = features.to(device), labels.to(device)

                # 勾配のリセット
                optimizer.zero_grad()

                # モデルの出力を計算
                outputs = model(features)

                # 出力とラベルの形状を整える
                b_size, f_size = outputs.size()
                # outputs = outputs.view(b_size, dim_out)

                # lossを計算
                loss = criterion(outputs, labels)

                # train phaseなら逆伝播する
                if phase == "train":
                    loss.backward()  # 勾配を計算
                    optimizer.step()  # パラメータを更新

                hyp = np.array(torch.argmax(outputs, 1).cpu())
                # ref = torch.argmax(labels, 1)
                ref = np.array(labels.cpu())

                # valid phaseなら推論結果を正解含めて出力する
                if phase == "valid":
                    valid_file = os.path.join(log_dir, str(epoch) + "_valid.log")
                    with open(valid_file, mode="a") as f:
                        f.write(str(outputs))
                        f.write("\n")
                        f.write(str(labels))
                        f.write("\n")
                        f.write(str(hyp == ref))
                        f.write("\n")

                batch_error = (hyp != ref).sum()
                total_loss += loss.item()
                total_utt += b_size
                total_error += (hyp != ref).sum()
                total_frames += len(ref)

                if phase == "train":
                    valid_file = os.path.join(log_dir, str(epoch) + "_train.log")
                    with open(valid_file, mode="a") as f:
                        f.write(str(hyp))
                        f.write("\n")
                        f.write(str(ref))
                        f.write("\n")
                        f.write(str(hyp == ref))
                        f.write("\n")

                # 1エポック終了

            # 損失の累積を発話数で割る
            epoch_loss = total_loss / total_utt
            epoch_error = 100.0 * total_error / total_frames
            print(f"{phase}")
            print(f"epoch {epoch} -> epoch_loss : {epoch_loss}")
            print(f"epoch {epoch} -> epoch_error: {epoch_error}")
            wandb.log({"loss": epoch_loss})
            wandb.log({"error": epoch_error})

            # valid特有の処理
            if phase == "valid":
                if epoch == 0 or best_loss > epoch_loss:
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), output_dir + "/best_model.pt")
                    best_epoch = epoch
                    counter_for_ES = 0
                else:
                    if epoch + 1 > lr_decay_start_epoch:
                        if counter_for_ES + 1 >= early_stop_threshold:
                            early_stop_flag = True
                        else:
                            # 学習率を減衰
                            if lr_decay_factor < 1.0:
                                for i, param_group in enumerate(optimizer.param_groups):
                                    if i == 0:
                                        lr = param_group["lr"]
                                        dlr = lr_decay_factor * lr
                                    param_group["lr"] = dlr
                            # counter_for_ES += 1
            print(f"epoch {epoch} -> best_epoch: {best_epoch}")
    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pt"))
    return


def setup(config: DictConfig):
    # set RandomSeed
    seed_everything(config.seed)

    # create directory
    os.makedirs(config.train.exp_dir, exist_ok=True)  # output
    os.makedirs(config.train.out_dir, exist_ok=True)  # output
    os.makedirs(config.logger.save_dir, exist_ok=True)  # logger

    # DataModule
    datamodule = AudioTextDataModule(config)
    datamodule.setup("")

    # callbacks
    callbacks = []
    for _, cb_conf in config.callbacks.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            # log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    # logger
    logger = hydra.utils.instantiate(config.logger)

    # criterion
    criterion = hydra.utils.instantiate(config.criterion)

    # optimizer
    optimizer = hydra.utils.instantiate(config.optimizer)

    # scheduler 将来的には設定する予定
    scheduler = None

    # model
    model = hydra.utils.instantiate(config.model, criterion=criterion, optimizer=optimizer, scheduler=scheduler)

    return model, datamodule, callbacks, logger


@hydra.main(config_path="conf/simpleSR", config_name="config", version_base="1.3")
def train(config: DictConfig):
    model, datamodule, callbacks, logger = setup(config)
    trainer = Trainer(
        callbacks=callbacks,
        logger=logger,
        max_epochs=config.train.max_epochs,
        devices=config.devices,
        accelerator="auto",
    )
    ckpt_path = config.ckpt_path
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


@hydra.main(config_path="conf/simpleSR", config_name="config", version_base="1.3")
def check_config(config: DictConfig):
    setup(config)
    print(OmegaConf.to_yaml(config))
    exit(0)


if __name__ == "__main__":
    # check_config()
    train()
