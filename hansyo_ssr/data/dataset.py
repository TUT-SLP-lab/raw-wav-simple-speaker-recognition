# -*- coding: utf-8 -*-

#
# Pytorchで用いるDatasetの定義
#

# sysモジュールをインポート
import sys

# 数値演算用モジュール(numpy)をインポート
import numpy as np

# PytorchのDatasetモジュールをインポート
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


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
        self.spk_ids_to_token = {}

        with open(data_config.spk, mode="r") as f_spks:
            spk_ids = [spk_id.strip() for spk_id in f_spks]
            spk_ids_len = len(spk_ids)
            for i, label in enumerate(spk_ids):
                self.spk_ids_to_token[label] = torch.Tensor([1.0 if i == idx else 0.0 for idx in range(spk_ids_len)])
        with open(phase_config.scp, mode="r") as f_feats:
            for line_feats in f_feats:
                # 発話IDと話者IDに分割
                id, spk_id = line_feats.split()
                # 発話IDを追加
                self.id_list.append(id)
                # 特徴量ファイルのパスを追加
                # XXX: 今回は、*-inference.npy か *-feats.scp という形で保存されている
                self.feat_list.append(f"{data_config.feats.dir}/{id}-feats.npy")
                # ラベル(番号で記載)を1hotなベクトルに変換する(今後のため)
                self.label_list.append(self.spk_ids_to_token[spk_id])
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
        feat = np.load(self.feat_list[idx])

        # 平均と標準偏差を使って正規化(標準化)を行う
        # feat = (feat - self.feat_mean) / self.feat_std

        # ラベル
        label = self.label_list[idx]

        # 発話ID
        utt_id = self.id_list[idx]

        # 特徴量，ラベル，フレーム数，
        # ラベル長，発話IDを返す
        return (torch.from_numpy(feat), label, utt_id)


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
    batch: [(feat, label, utt_id), ...]

    return
    ------
        feats: torch.Tensor (B x feat_length x feat_dim)
        labels: torch.Tensor (B x label_length x label_dim)
        feat_lens: torch.Tensor (B)
        label_lens: torch.Tensor (B)
        utt_ids: list("str") (B)
    """
    feats = pad_sequence([t[0] for t in batch], batch_first=True)
    labels = pad_sequence([t[1] for t in batch], batch_first=True)
    feat_lens = torch.Tensor([len(t[0]) for t in batch])
    label_lens = torch.Tensor([len(t[1]) for t in batch])
    utt_ids = [t[2] for t in batch]
    return feats, labels, feat_lens, label_lens, utt_ids
