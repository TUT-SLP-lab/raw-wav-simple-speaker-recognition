# import torch
# import numpy as np
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# ミニバッチには、DataLoaderを使えば良いらしい？(後で実装すること)
class simpleSR(nn.Module):
    def __init__(self, hidden_dim=4096, fc1_params=2058, fc2_params=512, output_nums=3):
        super(simpleSR, self).__init__()
        # パラメータ値を指定
        self.hidden_dim = hidden_dim  # LSTMの隠れ層の次元数(LSTMが出力するベクトルの次元数)
        self.fc1_params = fc1_params
        self.fc2_params = fc2_params
        self.output_nums = output_nums  # 出力の次元数

        # モデルの層を準備
        self.lstm = nn.LSTM(1, self.hidden_dim, batch_first=True)  # Raw-wavの場合、1フレームに付き1つの値を入力する?
        # とりあえず、DropOutはなし
        self.fc1 = nn.Linear(self.hidden_dim, self.fc1_params)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.fc1_params, self.fc2_params)
        self.relu2 = nn.ReLU()
        self.softmax = nn.Softmax(self.output_nums)

    def forward(self, sequence, lengths):
        ''' ネットワーク計算
        sequence: 各発話の入力系列 [B x T x D]
        length: 各発話の系列長 [B]
            B: ミニバッチの初話数(ミニバッチサイズ)
            T: テンソルの系列長(パディングを含む)
            D: 入力次元数
        '''
        output = sequence
        output_length = lengths
        # データをネットワークに入力
        # 入力をPackedSequenceデータに変換(既にパディングされているという前提)
        output = pack_padded_sequence(output, output_length, batch_first=True)
        self.lstm.flatten_parameters()  # GPUとcuDNNを使う際に処理が早くなる
        output, (h, c) = self.lstm(output)  # LSTMを計算
        # PackedSequenceからTensorに戻す
        output = pad_packed_sequence(output, batch_first=True)
        output = self.relu1(self.fc1(output))
        output = self.relu2(self.fc2(output))
        output = self.softmax(output)
        return output


# モデルの概形を出力する
if __name__ == "__main__":
    network = simpleSR()
    print(network)
