# import torch
import numpy as np
from torch import nn
import pytorch_lightning as pl
# from torch.nn.utils.rnn import pack_padded_sequence
# from torch.nn.utils.rnn import pad_packed_sequence


def calc_frame_len(frame_num, window_size, stride_size):
    return int(np.floor(((frame_num + (1 - window_size) - 1) / stride_size) + 1))
    return int(np.ceil((frame_num - (window_size - stride_size)) / stride_size))


# TODO: hydraを使ってパラメータを管理
# ミニバッチには、DataLoaderを使えば良いらしい？(後で実装すること)
class simpleSR(pl.LightningModule):
    def __init__(self,
                 feats_dim,
                 input_length=8192,
                 input_channel=80,
                 conv_nums=3,
                 fc1_params=2048,
                 fc2_params=512,
                 output_nums=3):
        super(simpleSR, self).__init__()
        # パラメータ値を指定
        self.conv_nums = conv_nums
        self.input_length = input_length
        self.input_channel = input_channel
        # self.hidden_dim = hidden_dim  # LSTMの隠れ層の次元数(LSTMが出力するベクトルの次元数)
        self.fc1_params = fc1_params
        self.fc2_params = fc2_params
        self.output_nums = output_nums  # 出力の次元数

        # モデルの層を準備
        # self.lstm = nn.LSTM(self.input_length, self.hidden_dim, batch_first=True)  #k Raw-wavの場合、1フレームに付き1つの値を入力する?
        self.conv_kernel_sizes = reversed([(2 * i + 3) for i in range(0, self.conv_nums)])
        self.conv_stride = 1
        # self.pool_kernel_size = (3, int(feats_dim))
        self.pool_kernel_size = (3)
        self.pool_stride = 2
        self.convs = []
        frame_num = self.input_length
        output_channel = self.input_channel
        for kernel in self.conv_kernel_sizes:
            self.input_channel = output_channel
            output_channel = self.input_channel * 4
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(self.input_channel, output_channel, kernel_size=kernel, stride=self.conv_stride),
                    nn.BatchNorm1d(output_channel),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(kernel_size=self.pool_kernel_size, stride=self.pool_stride)
                )
            )
            # conv_f_num = self.flatten_size(frame_num, kernel, self.conv_stride)
            # frame_num = self.flatten_size(conv_f_num, self.pool_kernel_size, self.pool_stride)
            conv_f_num = calc_frame_len(frame_num, kernel, self.conv_stride)
            frame_num = calc_frame_len(conv_f_num, self.pool_kernel_size, self.pool_stride)
            print(conv_f_num, frame_num, frame_num * output_channel)
        # リストで定義するとうまく学習できないので、モデルを特定の形式にする
        self.convs = nn.ModuleList(self.convs)
        # とりあえず、DropOutはなし
        self.flatten_size = frame_num * output_channel
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(nn.Linear(self.flatten_size, self.fc1_params), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(self.fc1_params, self.fc2_params), nn.ReLU())
        self.fc_output = nn.Sequential(nn.Linear(self.fc2_params, self.output_nums), nn.ReLU())
        self.softmax = nn.Softmax(dim=1)  # 各発話に対してSoftmaxをかける

    def forward(self, sequence):
        ''' ネットワーク計算
        sequence: 各発話の入力系列 [B x T x D]
        length: 各発話の系列長 [B]
            B: ミニバッチの初話数(ミニバッチサイズ)
            T: テンソルの系列長(パディングを含む)
            D: 入力次元数
        '''
        output = sequence
        for conv in self.convs:
            output = conv(output)
        output = self.flatten(output)  # 1列に並べる
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc_output(output)
        output = self.softmax(output)
        return output

    # TODO: Pytorch Lightningを使って動かせるようにする

    def _model_step():
        """ モデルの学習・検証において共通の処理の切り出し """
        pass

    def training_step():
        """ Pytorch Lightningの学習の1ステップ
        """
        pass

    def validation_step():
        """ Pytorch Lightningの検証の1ステップ
        """
        pass

    def test_step():
        """ Pytorch Lightningのテストの1ステップ
        """
        pass

    def configure_optimizers():
        """ Pytorch LightningのオプティマイザーとLRスケジューラーの設定
        """
        pass


# モデルの概形を出力する
if __name__ == "__main__":
    network = simpleSR(feats_dim=80, input_length=256)
    print(network)

    import torch
    from torchviz import make_dot
    data = torch.randn((1, 80, 256))
    y = network(data)
    image = make_dot(y, params=dict(network.named_parameters()), show_attrs=True, show_saved=True)
    # image = make_dot(y, params=dict(network.named_parameters()), show_attrs=True)
    image.format = "pdf"
    image.render("SimplSR")
