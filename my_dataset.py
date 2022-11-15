# -*- coding: utf-8 -*-

#
# Pytorchで用いるDatasetの定義
#

# PytorchのDatasetモジュールをインポート
import torch
from torch.utils.data import Dataset

# 数値演算用モジュール(numpy)をインポート
import numpy as np

# sysモジュールをインポート
import sys


class SequenceDataset(Dataset):
    ''' ミニバッチデータを作成するクラス
        torch.utils.data.Datasetクラスを継承し，
        以下の関数を定義する
        __len__: 総サンプル数を出力する関数
        __getitem__: 1サンプルのデータを出力する関数
    feat_scp:  特徴量リストファイル
    label_scp: ラベルファイル
    feat_mean: 特徴量の平均値ベクトル
    feat_std:  特徴量の次元毎の標準偏差を並べたベクトル
    pad_index: バッチ化の際にフレーム数を合わせる
               ためにpaddingする整数値
    splice:    前後(splice)フレームを特徴量を結合する
               splice=1とすると，前後1フレーム分結合
               するので次元数は3倍になる．
               splice=0の場合は何もしない
    '''

    def __init__(self,
                 feat_scp,
                 label_scp,
                 feat_mean,
                 feat_std,
                 input_length,
                 pad_index=0,
                 splice=0):
        # 発話の数
        self.num_utts = 0
        # 各発話のID
        self.id_list = []
        # 各発話の特徴量ファイルへのパスを記したリスト
        self.feat_list = []
        # 各発話の特徴量フレーム数を記したリスト
        self.feat_len_list = []
        # 特徴量の平均値ベクトル
        self.feat_mean = feat_mean
        # 特徴量の標準偏差ベクトル
        self.feat_std = feat_std
        # 標準偏差のフロアリング
        # (0で割ることが発生しないようにするため)
        self.feat_std[self.feat_std < 1E-10] = 1E-10
        # 特徴量の次元数
        self.feat_dim = np.size(self.feat_mean)
        # 各発話のラベル
        self.label_list = []
        # 各発話のラベルの長さを記したリスト
        self.label_len_list = []
        # フレーム数の最大値
        self.max_feat_len = 0
        # ラベル長の最大値
        self.max_label_len = 0
        # フレーム埋めに用いる整数値
        self.pad_index = pad_index
        # splice:前後nフレームの特徴量を結合
        self.splice = splice
        # window:特徴量中の学習に用いるフレーム数
        self.input_length = input_length

        # 特徴量リスト，ラベルを1行ずつ
        # 読み込みながら情報を取得する
        with open(feat_scp, mode='r') as file_f, \
                open(label_scp, mode='r') as file_l:
            for (line_feats, line_label) in zip(file_f, file_l):
                # 各行をスペースで区切り，
                # リスト型の変数にする
                parts_feats = line_feats.split()
                parts_label = line_label.split()

                # 発話ID(partsの0番目の要素)が特徴量と
                # ラベルで一致していなければエラー
                if parts_feats[0] != parts_label[0]:
                    sys.stderr.write('IDs of feat and '
                                     'label do not match.\n')
                    exit(1)

                # 発話IDをリストに追加
                self.id_list.append(parts_feats[0])
                # 特徴量ファイルのパスをリストに追加
                self.feat_list.append(parts_feats[1])
                # フレーム数をリストに追加
                feat_len = np.int64(parts_feats[2])
                self.feat_len_list.append(feat_len)

                # ラベル(番号で記載)をint型のnumpy arrayに変換
                # 話者認識のため、1hotなベクトルにする必要がある
                # -> 実装の感じを見るにその必要はなさそう
                idx = np.int64(parts_label[1])
                # label = np.array([(1 if i == idx else 0) for i in range(3)], dtype=np.long)
                label = idx  # labelは正解そのもので良い
                # ラベルリストに追加
                self.label_list.append(label)
                # ラベルの長さを追加 これは話者認識では実際には使わない。消しても良いかも
                # self.label_len_list.append(len(label))
                self.label_len_list.append(1)

                # 発話数をカウント
                self.num_utts += 1

        # フレーム数の最大値を得る
        self.max_feat_len = np.max(self.feat_len_list)
        # ラベル長の最大値を得る
        self.max_label_len = np.max(self.label_len_list)

        # ラベルデータの長さを最大フレーム長に
        # 合わせるため，pad_indexの値で埋める
        for n in range(self.num_utts):
            # 埋めるフレームの数
            # = 最大フレーム数 - 自分のフレーム数
            pad_len = self.max_label_len - self.label_len_list[n]
            # pad_indexの値で埋める
            self.label_list[n] = np.pad(self.label_list[n],
                                        [0, pad_len],
                                        mode='constant',
                                        constant_values=self.pad_index)

    def __len__(self):
        ''' 学習データの総サンプル数を返す関数
        本実装では発話単位でバッチを作成するため，
        総サンプル数=発話数である．
        '''
        return self.num_utts

    def __getitem__(self, idx):
        ''' サンプルデータを返す関数
        本実装では発話単位でバッチを作成するため，
        idx=発話番号である．
        '''
        # 特徴量系列のフレーム数
        feat_len = self.feat_len_list[idx]
        # ラベルの長さ
        label_len = self.label_len_list[idx]

        # 特徴量データを特徴量ファイルから読み込む
        feat = np.fromfile(self.feat_list[idx],
                           dtype=np.float32)
        #                  dtype=np.float16)

        # フレーム数 x 次元数の配列に変形
        if (self.feat_dim != 1):
            feat = feat.reshape(-1, self.feat_dim)
        else:
            feat = feat.reshape(self.feat_dim, -1)

        # 平均と標準偏差を使って正規化(標準化)を行う
        feat = (feat - self.feat_mean) / self.feat_std

        # featを固定長に変換
        _start_point = torch.randint(0, high=(feat_len - self.input_length), size=(1, 1))[0]
        feat = feat[_start_point:_start_point + self.input_length]
        feat = feat.T

        # ラベル
        label = self.label_list[idx]

        # 発話ID
        utt_id = self.id_list[idx]

        # 特徴量，ラベル，フレーム数，
        # ラベル長，発話IDを返す
        return (feat, label, feat_len, label_len, utt_id)
