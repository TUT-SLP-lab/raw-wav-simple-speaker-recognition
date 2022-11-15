import torch
import os
import sys
import numpy as np
from my_dataset import SequenceDataset
from torch.utils.data import DataLoader
from simpleSR import simpleSR

if __name__ == "__main__":
    # 各種学習データのパス
    # 学習データ等の特徴量(feats.scp)が存在するディレクトリ
    feat_dir_train = 'data/mfcc/train'
    feat_dir_dev = 'data/mfcc/dev'
    feat_dir_eval = 'data/mfcc/eval'

    # 実験ディレクトリ
    # train_set_name = 'train_small' or 'train_large'
    train_set_name = os.path.basename(feat_dir_train)
    exp_dir = './exp_' + os.path.basename(feat_dir_train)

    # 学習/開発データの特徴量リストファイル
    feat_scp_eval = os.path.join(feat_dir_eval, 'feats.scp')

    # 学習/開発データのラベルファイル
    label_eval = "data/eval/label"

    # モデルパラメータ
    dnn_file = './exp_train/simple-SR/best_model.pt'

    # トークンリスト
    token_list_path = os.path.join(exp_dir, 'data', 'token_list')
    # 学習結果を出力するディレクトリ
    output_dir = os.path.join(exp_dir, 'simple-SR')

    # モデルパラメータ
    batch_size = 90
    max_num_epoch = 50
    hidden_dim = 4096  # 隠れ層の次元数
    fc1_params = 2048
    fc2_params = 512
    splice = 0  # 何だったか忘れたけど、0で良かったことだけは覚えてる
    initial_learning_rate = 0.01  # 開始時の学習率
    lr_decay_start_epoch = 10  # 学習率の玄瑞やESの判定を始めるエポック数
    lr_decay_factor = 0.5  # 学習率の減衰率
    early_stop_threshold = 3  # ESするのに最低損失が何epoch変化しないかを判断する

    # 入力データのパラメータ
    feats_dim = 80  # MFCCのbin数

    # train, dev, evalのデータセットを作る
    datasets = []
    feat_means = []
    feat_stds = []

    for mean_path, label_path, feat_path in zip([feat_dir_eval], [label_eval], [feat_scp_eval]):
        # 訓練データから計算された特徴量の平均/標準偏差ファイル
        mean_std_file = os.path.join(mean_path, 'mean_std.txt')

        # 特徴量を読み込む
        # 特徴量の平均/標準偏差ファイルを読み込む
        with open(mean_std_file, mode='r') as f:
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
            feat_means.append(np.array(feat_mean,
                                       dtype=np.float32))
            feat_stds.append(np.array(feat_std,
                                      dtype=np.float32))
        datasets.append(SequenceDataset(
            feat_path,
            label_path,
            feat_means[-1],
            feat_stds[-1]))

    # 次元数の情報を得る
    feat_dim = np.size(feat_means[0])

    # NNを構築する
    dim_out = 3
    model = simpleSR(feat_dim, hidden_dim, fc1_params, fc2_params, dim_out)
    model.load_state_dict(torch.load(dnn_file))

    # CUDAが使える場合はモデルパラメータをGPUに置く
    if torch.cuda.is_available():
        device_gpu = torch.device('cuda')
    device_cpu = torch.device('cpu')
    model = model.to(device_gpu)

    # modelを推論モードに設定
    model.eval()

    eval_loader = DataLoader(datasets[0],
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=8)
    for (features, labels, feat_len, label_len, utt_ids) in eval_loader:
        features = features.to(device_gpu)
        outputs = model(features, feat_len)
        outputs = outputs.to(device_cpu)
        outputs = outputs.detach().numpy()
        labels = labels.detach().numpy()
        for idx, label, output in zip(utt_ids, labels, outputs):
            if(np.argmax(output) != 2):
                print(f"{idx} {label} {output} {np.argmax(label)} {np.argmax(output)}")
