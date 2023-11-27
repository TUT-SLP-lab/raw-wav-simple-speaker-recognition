from typing import Any

import torch
from lightning import LightningModule
from torch.nn import LSTM, CrossEntropyLoss, Linear, ReLU, Sequential, Softmax
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torcheval.metrics import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)


class simpleSR(LightningModule):
    def __init__(
        self,
        input_channel: int = 80,
        hidden_dim: int = 1024,
        fc1_params: int = 2048,
        fc2_params: int = 512,
        output_nums: int = 2,
        criterion: Any = None,
        accuracy: Any = None,
        precision: Any = None,
        recall: Any = None,
        f1_score: Any = None,
        optimizer: Optimizer = AdamW,
        scheduler: LRScheduler = None,
    ):
        super(simpleSR, self).__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(ignore=["criterion"])

        # モデルの層を準備
        self.lstm = LSTM(input_channel, hidden_dim, batch_first=True)
        # self.fc1 = Sequential(Linear(hidden_dim, fc1_params), ReLU())
        # self.fc2 = Sequential(Linear(fc1_params, fc2_params), ReLU())
        # self.fc_output = Sequential(Linear(fc2_params, output_nums), ReLU())
        self.fc_output = Sequential(Linear(hidden_dim, output_nums))

        # 推論のためのSoftMax
        self.inference_softmax = Softmax(dim=1)

        self.criterion = criterion if criterion is not None else CrossEntropyLoss()

        # ロギングのための初期化 -> on_fit_startで実体の生成
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score

    def forward(self, sequence, seq_lens):
        """
        ネットワーク計算

        Params
        ---
        sequence: 各発話の入力系列 [B x T x D]
        length: 各発話の系列長 [B]
            B: ミニバッチの初話数(ミニバッチサイズ)
            T: テンソルの系列長(パディングを含む)
            D: 入力次元数
        """
        # パディング済みのTensorをPackedSequenceに変換する
        output = pack_padded_sequence(sequence, seq_lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (_, _) = self.lstm(output)  # 最終層の出力を話者特徴量とする
        # PackedSequence -> padded Tensor
        output, input_size = pad_packed_sequence(packed_output, batch_first=True)
        # (N, *, hidden_dim) -> (N, hidden_dim) の形に、LSTMの最終出力だけを取り出す
        output = output[:, -1, :]
        # output = self.fc1(output)  # LSTM層の出力をプロジェクション?
        # output = self.fc2(output)  # 話者特徴の学習: この層の出力が、このネットワークで得られる話者特徴量となる
        output = self.fc_output(output)  # 特徴量をもとに話者分類を行う
        return output

    ########################################
    #                Utils                 #
    ########################################

    def _model_step(self, batch):
        """モデルの学習・検証において共通の処理の切り出し"""
        feat, y, feat_lens, _, _ = batch
        y_hat = self.forward(feat, feat_lens)
        loss = self.criterion(y_hat, y)
        # update data
        y_label = y.argmax(dim=1).to(y_hat.device)
        self.accuracy.update(y_hat, y_label)
        self.precision.update(y_hat, y_label)
        self.recall.update(y_hat, y_label)
        self.f1_score.update(y_hat, y_label)
        return (loss, y_hat)

    def _log(self, label, val, batch_size, on_step=True, on_epoch=True):
        self.log(
            label,
            val,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )

    def _epoch_end(self, phase: str, average_loss):
        # logging
        self.log(f"{phase}/average_loss", average_loss, on_step=False, on_epoch=True, logger=True)
        self.log(f"{phase}/accuracy", self.accuracy.compute(), on_step=False, on_epoch=True, logger=True)
        self.log(f"{phase}/precision", self.precision.compute(), on_step=False, on_epoch=True, logger=True)
        self.log(f"{phase}/recall", self.recall.compute(), on_step=False, on_epoch=True, logger=True)
        self.log(f"{phase}/f1_score", self.f1_score.compute(), on_step=False, on_epoch=True, logger=True)
        # reset data
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1_score.reset()

    ########################################
    #          Pytorch Lightning           #
    ########################################
    def on_fit_start(self):
        self.accuracy = self.accuracy() if self.accuracy is not None else MulticlassAccuracy(device=self.device)
        self.precision = self.precision() if self.precision is not None else MulticlassPrecision(device=self.device)
        self.recall = self.recall() if self.recall is not None else MulticlassRecall(device=self.device)
        self.f1_score = self.f1_score() if self.f1_score is not None else MulticlassF1Score(device=self.device)

    def training_step(self, batch, batch_idx):
        batch_size = len(batch[2])
        loss, _ = self._model_step(batch)
        self.training_step_outputs.append(loss)
        self._log("train/loss", loss, batch_size)
        return {"loss": loss}

    def on_training_epoch_end(self):
        self._epoch_end("train", torch.stack(self.training_step_outputs).mean())
        self.training_step_outputs.clear()  # free memory

    def validation_step(self, batch, batch_idx):
        batch_size = len(batch[2])
        loss, _ = self._model_step(batch)
        self.validation_step_outputs.append(loss)
        self._log("val/loss", loss, batch_size)
        return {"loss": loss}

    def on_validation_epoch_end(self):
        self._epoch_end("val", torch.stack(self.validation_step_outputs).mean())
        self.validation_step_outputs.clear()  # free memory

    def test_step(self, batch, batch_idx):
        batch_size = len(batch[2])
        loss, pred = self._model_step(batch)
        self.test_step_outputs.append(loss)
        self._log("test/loss", loss, batch_size)
        return {"loss": loss}

    def on_test_epoch_end(self):
        self._epoch_end("test", torch.stack(self.test_step_outputs).mean())
        self.test_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


# モデルの概形を出力する
if __name__ == "__main__":
    network = simpleSR()
    print(network)

    import torch
    from torchviz import make_dot

    data = torch.randn((1, 80, 256))
    y = network(data)
    image = make_dot(y, params=dict(network.named_parameters()), show_attrs=True, show_saved=True)
    # image = make_dot(y, params=dict(network.named_parameters()), show_attrs=True)
    image.format = "pdf"
    image.render("SimplSR")
