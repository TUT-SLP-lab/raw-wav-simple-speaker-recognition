from typing import Any

# import torch
import numpy as np
from lightning import LightningModule
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torcheval.metrics import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)


# TODO: 2話者にした対応を行う必要がある
class X_vector_lightning(LightningModule):
    def __init__(
        self,
        x_vector_model,
        output_nums: int = 2,
        criterion_a: Any = None,
        criterion_b: Any = None,
        criterion_distance: Any = None,
        accuracy: Any = None,
        precision: Any = None,
        recall: Any = None,
        f1_score: Any = None,
        optimizer: Optimizer = AdamW,
        scheduler: LRScheduler = None,
        inference_out_dir: str = None,
    ):
        super(X_vector_lightning, self).__init__()
        self.save_hyperparameters(
            ignore=[
                "x_vector_model",
                "criterion_a",
                "criterion_b",
                "criterion_distance",
            ]
        )
        self.model = x_vector_model

        self.criterion_a = criterion_a if criterion_a is not None else CrossEntropyLoss()
        self.criterion_b = criterion_b if criterion_b is not None else CrossEntropyLoss()
        self.criterion_distance = criterion_distance if criterion_distance is not None else MSELoss()

        self.inference_out_dir = inference_out_dir

    def forward(self, sequence_a, sequence_b):
        return self.model.forward(sequence_a, sequence_b)

    ########################################
    #                Utils                 #
    ########################################

    def _model_step(self, batch, phase):
        """モデルの学習・検証において共通の処理の切り出し"""
        # feat, y, feat_lens, _, _ = batch
        (
            feats_a,
            feats_b,
            labels_a,
            labels_b,
            spk_dists,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = batch

        # y_hat, _, _ = self.forward(feat)
        (
            (voice_a_output, _, voice_a_x_vec),
            (voice_b_output, _, voice_b_x_vec),
            voice_distance,
        ) = self.forward(feats_a, feats_b)

        voice_a_loss = self.criterion_a(voice_a_output, labels_a)
        voice_b_loss = self.criterion_b(voice_b_output, labels_b)
        distance_loss = self.criterion_distance(voice_distance, spk_dists)
        loss = voice_a_loss + voice_b_loss + distance_loss

        # update data
        # y_label = y.argmax(dim=1).to(y_hat.device)
        y_label_a = labels_a.argmax(dim=1).to(voice_a_output.device)
        y_label_b = labels_b.argmax(dim=1).to(voice_b_output.device)
        self._update_evaluation_metrics(phase, voice_a_output, y_label_a, voice_b_output, y_label_b)

        return (
            (loss, voice_a_loss, voice_b_loss, distance_loss),
            (voice_a_output, voice_b_output),
            (voice_a_x_vec, voice_b_x_vec),
        )

    def _update_evaluation_metrics(self, phase, y_hat_a, y_label_a, y_hat_b, y_label_b):
        self.accuracy[phase].update(y_hat_a, y_label_a)
        self.accuracy[phase].update(y_hat_b, y_label_b)
        self.precision[phase].update(y_hat_a, y_label_a)
        self.precision[phase].update(y_hat_b, y_label_b)
        self.recall[phase].update(y_hat_a, y_label_a)
        self.recall[phase].update(y_hat_b, y_label_b)
        self.f1_score[phase].update(y_hat_a, y_label_a)
        self.f1_score[phase].update(y_hat_b, y_label_b)

    def _step_log(self, label, val, batch_size, on_step=True, on_epoch=True):
        self.log(
            label,
            val,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=True,
            batch_size=batch_size,
        )

    def _epoch_end(self, phase: str):
        # logging
        self.log(f"{phase}/accuracy", self.accuracy[phase].compute(), on_step=False, on_epoch=True)
        self.log(f"{phase}/precision", self.precision[phase].compute(), on_step=False, on_epoch=True)
        self.log(f"{phase}/recall", self.recall[phase].compute(), on_step=False, on_epoch=True)
        self.log(f"{phase}/f1_score", self.f1_score[phase].compute(), on_step=False, on_epoch=True)
        # reset data
        self.accuracy[phase].reset()
        self.precision[phase].reset()
        self.recall[phase].reset()
        self.f1_score[phase].reset()

    def _save_xvecs(self, batch, xvecses):
        """Save xvecs to numpy file"""
        _, _, _, _, _, _, _, _, _, utt_ids_a, utt_ids_b = batch
        for utt_ids, xvecs in zip([utt_ids_a, utt_ids_b], xvecses):
            for utt_id, xvec in zip(utt_ids, xvecs):
                np.save(f"{self.inference_out_dir}/{utt_id}.npy", xvec.cpu().numpy())

    ########################################
    #          Pytorch Lightning           #
    ########################################
    def on_fit_start(self):
        self.accuracy = {}
        self.precision = {}
        self.recall = {}
        self.f1_score = {}

        if self.hparams.accuracy is not None:
            self.accuracy["train"] = self.hparams.accuracy()
            self.accuracy["val"] = self.hparams.accuracy()
            self.accuracy["test"] = self.hparams.accuracy()
        else:
            self.accuracy["train"] = MulticlassAccuracy(device=self.device)
            self.accuracy["val"] = MulticlassAccuracy(device=self.device)
            self.accuracy["test"] = MulticlassAccuracy(device=self.device)

        if self.hparams.precision is not None:
            self.precision["train"] = self.hparams.precision()
            self.precision["val"] = self.hparams.precision()
            self.precision["test"] = self.hparams.precision()

        else:
            self.precision["train"] = MulticlassPrecision(device=self.device)
            self.precision["val"] = MulticlassPrecision(device=self.device)
            self.precision["test"] = MulticlassPrecision(device=self.device)

        if self.hparams.recall is not None:
            self.recall["train"] = self.hparams.recall()
            self.recall["val"] = self.hparams.recall()
            self.recall["test"] = self.hparams.recall()
        else:
            self.recall["train"] = MulticlassRecall(device=self.device)
            self.recall["val"] = MulticlassRecall(device=self.device)
            self.recall["test"] = MulticlassRecall(device=self.device)

        if self.hparams.f1_score is not None:
            self.f1_score["train"] = self.hparams.f1_score()
            self.f1_score["val"] = self.hparams.f1_score()
            self.f1_score["test"] = self.hparams.f1_score()
        else:
            self.f1_score["train"] = MulticlassF1Score(device=self.device)
            self.f1_score["val"] = MulticlassF1Score(device=self.device)
            self.f1_score["test"] = MulticlassF1Score(device=self.device)

    def on_test_start(self):
        self.accuracy = {}
        self.precision = {}
        self.recall = {}
        self.f1_score = {}

        if self.hparams.accuracy is not None:
            self.accuracy["train"] = self.hparams.accuracy()
            self.accuracy["val"] = self.hparams.accuracy()
            self.accuracy["test"] = self.hparams.accuracy()
        else:
            self.accuracy["train"] = MulticlassAccuracy(device=self.device)
            self.accuracy["val"] = MulticlassAccuracy(device=self.device)
            self.accuracy["test"] = MulticlassAccuracy(device=self.device)

        if self.hparams.precision is not None:
            self.precision["train"] = self.hparams.precision()
            self.precision["val"] = self.hparams.precision()
            self.precision["test"] = self.hparams.precision()

        else:
            self.precision["train"] = MulticlassPrecision(device=self.device)
            self.precision["val"] = MulticlassPrecision(device=self.device)
            self.precision["test"] = MulticlassPrecision(device=self.device)

        if self.hparams.recall is not None:
            self.recall["train"] = self.hparams.recall()
            self.recall["val"] = self.hparams.recall()
            self.recall["test"] = self.hparams.recall()
        else:
            self.recall["train"] = MulticlassRecall(device=self.device)
            self.recall["val"] = MulticlassRecall(device=self.device)
            self.recall["test"] = MulticlassRecall(device=self.device)

        if self.hparams.f1_score is not None:
            self.f1_score["train"] = self.hparams.f1_score()
            self.f1_score["val"] = self.hparams.f1_score()
            self.f1_score["test"] = self.hparams.f1_score()
        else:
            self.f1_score["train"] = MulticlassF1Score(device=self.device)
            self.f1_score["val"] = MulticlassF1Score(device=self.device)
            self.f1_score["test"] = MulticlassF1Score(device=self.device)

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._model_step(batch, "train")
        batch_size = len(batch[-1])
        self._step_log("train/loss", loss[0], batch_size=batch_size)
        self._step_log("train/loss_voice_a", loss[1], batch_size=batch_size)
        self._step_log("train/loss_voice_b", loss[2], batch_size=batch_size)
        self._step_log("train/loss_distance", loss[3], batch_size=batch_size)
        return loss[0]

    def on_training_epoch_end(self):
        self._epoch_end("train")

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self._model_step(batch, "val")
        batch_size = len(batch[-1])
        self._step_log("val/loss", loss[0], batch_size=batch_size, on_step=False)
        self._step_log("val/loss_voice_a", loss[1], batch_size=batch_size, on_step=False)
        self._step_log("val/loss_voice_b", loss[2], batch_size=batch_size, on_step=False)
        self._step_log("val/loss_distance", loss[3], batch_size=batch_size, on_step=False)
        return loss

    def on_validation_epoch_end(self):
        self._epoch_end("val")

    def test_step(self, batch, batch_idx):
        loss, preds, x_vecs = self._model_step(batch, "train")
        batch_size = len(batch[-1])
        self._step_log("test/loss", loss[0], batch_size=batch_size, on_step=False)
        self._step_log("test/loss_voice_a", loss[1], batch_size=batch_size, on_step=False)
        self._step_log("test/loss_voice_b", loss[2], batch_size=batch_size, on_step=False)
        self._step_log("test/loss_distance", loss[3], batch_size=batch_size, on_step=False)
        self._save_xvecs(batch, x_vecs)
        return loss, preds, x_vecs

    def on_test_epoch_end(self):
        self._epoch_end("test")

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
    from hansyo_ssr.model.x_vector import X_vector

    network = X_vector_lightning(X_vector(input_dim=80, num_classes=101))
    print(network)

    # モデルの出力が複雑なので、torchvizの出力は諦める
    # from torchviz import make_dot
    #
    # data = torch.randn((1, 256, 80))
    # y = network(data, data)
    # image = make_dot(y, params=dict(network.named_parameters()), show_attrs=True, show_saved=True)
    # # image = make_dot(y, params=dict(network.named_parameters()), show_attrs=True)
    # image.format = "pdf"
    # image.render("x_vector_lightning")
