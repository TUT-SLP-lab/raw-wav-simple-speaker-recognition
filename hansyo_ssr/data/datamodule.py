from hansyo_ssr.data.dataset import SequenceDataset, collate_fn
from omegaconf import DictConfig
from lightning.pytorch import LightningDataModule as LDM
from torch.utils.data import DataLoader


class AudioTextDataModule(LDM):
    def __init__(self, config: DictConfig):
        super()

        self.data_config = config.data
        self.prepare_data_per_node = False
        self.allow_zero_length_dataloader_with_multiple_devices = True

        self.save_hyperparameters()

    # plのデータのダウンロードとかを司るやつ
    def prepare_data(self):
        pass

    # データセットのsetupを本来はここで行う
    def setup(self, stage: str):
        pass

    def train_dataloader(self):
        return DataLoader(
            SequenceDataset(
                self.data_config,
                self.data_config.train,
            ),
            num_workers=self.data_config.loader.num_workers,
            batch_size=self.data_config.loader.batch_size,
            collate_fn=collate_fn,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            SequenceDataset(
                self.data_config,
                self.data_config.dev,
            ),
            num_workers=self.data_config.loader.num_workers,
            batch_size=self.data_config.loader.batch_size,
            collate_fn=collate_fn,
            pin_memory=True,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            SequenceDataset(
                self.data_config,
                self.data_config.test,
            ),
            num_workers=self.data_config.loader.num_workers,
            batch_size=self.data_config.loader.batch_size,
            collate_fn=collate_fn,
            pin_memory=True,
            shuffle=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            SequenceDataset(
                self.data_config,
                self.data_config.test,
            ),
            num_workers=self.data_config.loader.num_workers,
            batch_size=self.data_config.loader.batch_size,
            collate_fn=collate_fn,
            pin_memory=True,
            shuffle=False,
        )
