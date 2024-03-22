import os
from shutil import copytree

import hydra
import torch
from lightning import Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf

from hansyo_ssr.data.datamodule import AudioTextDataModule


def setup(config: DictConfig):
    """Setup training

    Params
    ------
        config: hydra's DictConfig

    Returns
    -------
        model: < pytorch lightning model >
        datamodule: < pytorch lightning datamodule >
        callbacks: List[< pytorch lightning callback >]
        logger: < pytorch lightning logger >
    """
    # set RandomSeed
    seed_everything(config.seed)

    # create save directory
    os.makedirs(config.train.exp_dir, exist_ok=True)
    os.makedirs(config.train.out_dir, exist_ok=True)
    os.makedirs(config.logger.save_dir, exist_ok=True)

    # save config.yaml
    OmegaConf.save(config=config, f=os.path.join(config.train.exp_dir, "config.yaml"))

    # save labels
    save_data_dir = os.path.join(config.train.exp_dir, config.data.dir)
    os.makedirs(save_data_dir, exist_ok=True)
    copytree(config.data.train.dir, os.path.join(save_data_dir, "train"), symlinks=False)
    copytree(config.data.dev.dir, os.path.join(save_data_dir, "dev"), symlinks=False)
    copytree(config.data.test.dir, os.path.join(save_data_dir, "dir"), symlinks=False)

    # logger = hydra.utils.instantiate(config.logger)
    logger = None  # x-vectorの再計測をしたいだけなので、loggerは使わない

    # criterion = hydra.utils.instantiate(config.criterion)
    optimizer = hydra.utils.instantiate(config.optimizer)
    scheduler = None  # 将来的には設定する予定

    # 純正pytorchのモデル
    base_model = hydra.utils.instantiate(config.base_model)

    # pytorch lightningのモデル
    model = hydra.utils.instantiate(
        config.model,
        x_vector_model=base_model,
        # criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        inference_out_dir=config.train.out_dir,
    )

    datamodule = AudioTextDataModule(config)
    datamodule.setup("")

    callbacks = []
    for _, cb_conf in config.callbacks.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return model, datamodule, callbacks, logger


@hydra.main(config_path="conf/simpleSR", config_name="reconstruction", version_base="1.3")
def test(config: DictConfig):
    model, datamodule, callbacks, _ = setup(config)
    trainer = Trainer(
        callbacks=callbacks,
        # logger=logger,
        max_epochs=config.train.max_epochs,
        devices=config.devices,
        accelerator="auto",
    )
    ckpt_path = config.ckpt_path
    if ckpt_path is None:
        exit("ckpt_path is None")
    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    test()
