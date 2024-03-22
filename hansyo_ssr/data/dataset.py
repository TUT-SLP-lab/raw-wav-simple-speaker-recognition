# -*- coding: utf-8 -*-

#
# Pytorchで用いるDatasetの定義
#

# sysモジュールをインポート
# import sys

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

# from joblib import Parallel, delayed
# from tqdm.auto import tqdm


# TODO: Hydra
class SequenceDataset(Dataset):
    """ミニバッチデータを作成するクラス
    torch.utils.data.Datasetクラスを継承し，
    以下の関数を定義する
    __len__: 総サンプル数を出力する関数
    __getitem__: 1サンプルのデータを出力する関数
    """

    def __init__(self, data_config, phase_config):
        # 発話数
        self.num_utts = 0
        # 各発話のID
        self.id_list = []
        # 話者IDのリスト
        # 各発話の特徴量ファイルへのパスを記したリスト
        self.feat_list = []
        # 各発話の特徴量フレーム数を記したリスト
        self.feat_len_list = []
        # XXX: 今回は使わない
        # # 特徴量の平均値・標準偏差ベクトル
        # self.feat_mean, self.feat_std = get_feat_mean_stds(phase_config.feats.mean_std)  # feat_mean
        # # 標準偏差のフロアリング (0で割ることが発生しないようにするため)
        # self.feat_std[self.feat_std < 1e-10] = 1e-10
        # 特徴量の次元数
        self.feat_dim = data_config.feats.bins
        # 各発話のラベル
        self.label_list = []
        # 各発話のラベルの長さを記したリスト
        self.label_len_list = []
        # フレーム数の最大値
        self.max_feat_len = 0
        # ラベル長の最大値
        self.max_label_len = 0
        # 話者IDを1hotベクトルに変換する辞書
        # { "spkid": torch.Tensor } の形式
        self.spkid_to_token = dict()
        # 話者距離リスト
        self.spk_dist_list = []

        print("Generating spk_id_to_token...")
        # spk_id_to_tokenを作成
        with open(data_config.spk.file, mode="r") as f_spks, open(data_config.spk.to_id_file) as f_to_id:
            self.spkid_to_id = dict()
            self.id_to_spkid = dict()
            for line in f_to_id:
                spk_id, id = line.strip().split(":")
                self.spkid_to_id[spk_id] = int(id)
                self.id_to_spkid[int(id)] = spk_id
            spk_ids_len = len(self.spkid_to_id)
            eye = torch.eye(spk_ids_len)
            for line in f_spks:
                label = line.strip()
                self.spkid_to_token[label] = eye[self.spkid_to_id[label]].clone().detach()

        print("Generating dataset...")
        # 発話IDと話者IDのリストを作成
        with open(phase_config.scp, mode="r") as f_feats:
            # # NOTE: 大量のデータを扱うため、joblibを使って並列化
            # # どうも、結果をzip展開するためのループで2回回すことになるため、内部処理が軽い場合は効かないっぽい
            #
            # def _get_feat_len(line_feats):
            #     # 発話IDと話者IDに分割
            #     id_a, id_b, spk_id_a, spk_id_b, dist_spk = line_feats.strip().split()
            #     # 発話IDを追加
            #     id_list = [id_a, id_b]
            #     # 特徴量ファイルのパスを追加 NOTE: 今回は、*-inference.npy か *-feats.scp という形で保存されている
            #     feat_list = [
            #         f"{data_config.feats.dir}/{id_a}-feats.npy",
            #         f"{data_config.feats.dir}/{id_b}-feats.npy",
            #     ]
            #     # ラベル(番号で記載)を1hotなベクトルに変換する(今後のため)
            #     label_list = [self.spkid_to_token[spk_id_a], self.spkid_to_token[spk_id_b]]
            #     # 話者距離を追加
            #     spk_dist = float(dist_spk) / 9.0  # 現在のスケールが[0., 9.]なので、[0., 1.]にスケール変換
            #
            #     return id_list, feat_list, label_list, spk_dist
            #
            # results = Parallel(n_jobs=-3)(
            #     [delayed(_get_feat_len)(line_feats) for line_feats in f_feats]
            # )
            # self.id_list, self.feat_list, self.label_list, self.spk_dist_list = zip(*results)

            for i, line_feats in enumerate(f_feats.readlines()):
                if i + 1 % 100000 == 0:
                    print(f"{i+1} lines processed.")
                # 発話IDと話者IDに分割
                id_a, id_b, spk_id_a, spk_id_b, dist_spk = line_feats.strip().split()
                # 発話IDを追加
                self.id_list.append((id_a, id_b))
                # 特徴量ファイルのパスを追加 NOTE: 今回は、*-inference.npy か *-feats.scp という形で保存されている
                self.feat_list.append(
                    (f"{data_config.feats.dir}/{id_a}-feats.npy", f"{data_config.feats.dir}/{id_b}-feats.npy")
                )
                # ラベル(番号で記載)を1hotなベクトルに変換する(今後のため)
                self.label_list.append((self.spkid_to_token[spk_id_a], self.spkid_to_token[spk_id_b]))
                # NOTE: 1. 0.0 ~ 6.0 -> -1.0 ~ 1.0
                # self.spk_dist_list.append((float(dist_spk) -3.0) / 3.0)
                # NOTE: 2. 0.0 ~ 6.0 -> 0.0 ~ 1.0
                self.spk_dist_list.append(float(dist_spk) / 6)
        print("Done.")
        # 発話数をメモ
        self.num_utts = len(self.id_list)

    def __len__(self):
        """学習データの総サンプル数を返す関数
        本実装では発話単位でバッチを作成するため，
        総サンプル数=発話数である．
        """
        return self.num_utts

    def __getitem__(self, idx):
        """サンプルデータを返す関数
        本実装では発話単位でバッチを作成するため，
        idx=発話番号である．

        Return
        ------
            feat: torch.Tensor (feats_len x feat_dim)
            label: int
            utt_id: str
        """

        # 特徴量データを特徴量ファイルから読み込む
        feat_a = np.load(self.feat_list[idx][0])
        feat_b = np.load(self.feat_list[idx][1])

        # ラベル
        label_a = self.label_list[idx][0]
        label_b = self.label_list[idx][1]

        # 発話ID
        utt_id_a = self.id_list[idx][0]
        utt_id_b = self.id_list[idx][1]

        # spk_dist
        spk_dist = self.spk_dist_list[idx]

        # 特徴量，ラベル，フレーム数，
        # ラベル長，発話IDを返す
        return (
            torch.from_numpy(feat_a),
            torch.from_numpy(feat_b),
            label_a,
            label_b,
            spk_dist,
            utt_id_a,
            utt_id_b,
        )


def get_feat_mean_stds(mean_std_scp):
    """ファイルから平均と標準偏差を返す"""
    with open(mean_std_scp, mode="r") as f:
        # 全行読み込み
        lines = f.readlines()
        # 1行目(0始まり)が平均値ベクトル(mean)，
        mean_line = lines[1]
        # 3行目が標準偏差ベクトル(std)
        std_line = lines[3]
        # スペース区切りのリストをndarrayに変換
        feat_mean = np.array(mean_line.split(), dtype=np.float32)
        feat_std = np.array(std_line.split(), dtype=np.float32)
    return feat_mean, feat_std


def collate_fn(batch):
    """batchの形状を整える
    Input
    -----
    batch: [(feat_a, feat_b, label_a, label_b, spk_dist, utt_id_a, utt_id_b), ...]

    return
    ------
        feats: torch.Tensor (B x feat_length x feat_dim)
        labels: torch.Tensor (B x label_length x label_dim)
        feat_lens: torch.Tensor (B)
        label_lens: torch.Tensor (B)
        utt_ids: list("str") (B)
    """

    # NOTE: 複数の発話のパディングを等しくするため、
    # 一旦伸長してから、batch_first=Trueでパディングする
    # その後、aとbに分割することで、発話を分けつつ、パディングを行ったことにする
    feats = []
    labels = []
    feat_lens = []
    label_lens = []
    utt_ids = []
    spk_dists = []
    for feat_a, feat_b, label_a, label_b, spk_dist, utt_id_a, utt_id_b in batch:
        feats.append(feat_a)
        feats.append(feat_b)
        feat_lens.append(len(feat_a))
        feat_lens.append(len(feat_b))
        labels.append(label_a)
        labels.append(label_b)
        label_lens.append(len(label_a))
        label_lens.append(len(label_b))
        utt_ids.append(utt_id_a)
        utt_ids.append(utt_id_b)
        spk_dists.append(spk_dist)
    feats = pad_sequence([t for t in feats], batch_first=True)
    labels = pad_sequence([t for t in labels], batch_first=True)
    feat_lens = torch.Tensor(feat_lens)
    label_lens = torch.Tensor(label_lens)

    feats_a = feats[::2]
    feats_b = feats[1::2]
    labels_a = labels[::2]
    labels_b = labels[1::2]
    feat_lens_a = feat_lens[::2]
    feat_lens_b = feat_lens[1::2]
    label_lens_a = label_lens[::2]
    label_lens_b = label_lens[1::2]
    utt_ids_a = utt_ids[::2]
    utt_ids_b = utt_ids[1::2]
    spk_dists = torch.Tensor(spk_dists)

    return (
        feats_a,
        feats_b,
        labels_a,
        labels_b,
        spk_dists,
        feat_lens_a,
        feat_lens_b,
        label_lens_a,
        label_lens_b,
        utt_ids_a,
        utt_ids_b,
    )
