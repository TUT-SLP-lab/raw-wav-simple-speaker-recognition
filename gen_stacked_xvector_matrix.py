import argparse

import hydra
import numpy as np
import torch
from omegaconf import DictConfig


def load_data(path: str) -> torch.Tensor:
    data = np.load(path)
    data_tensor = torch.from_numpy(data)
    return data_tensor


def load_config(config: DictConfig) -> DictConfig:
    return config


@hydra.main(config_path="conf/simpleSR", config_name="mid_gen_config", version_base="1.3")
def main(config: DictConfig):
    if config.xvector_dir is None or config.xvector_dir == "":
        print("xvector_dir is not specified.")
        exit(1)
    xvector_dir = config.xvector_dir
    spk_list = []
    with open(config.data.spk.to_id_file, "r") as f:
        for line in f.readlines():
            name, id = line.strip().split(":")
            spk_list.append((name, int(id)))
    with open(config.data.spk.female.file, "r") as f:
        female_spks = f.read().splitlines()
    # stacked_xvector_matrix = torch.zeros((len(spk_list), 512))
    stacked_xvector_matrix = torch.zeros((len(spk_list), 64))
    for name, id in spk_list:
        if name in female_spks:
            stacked_xvector_matrix[id] = load_data(f"{xvector_dir}/{name}_mean.npy")
    np.save(f"{xvector_dir}/stacked_xvector_matrix.npy", stacked_xvector_matrix.numpy())


if __name__ == "__main__":
    main()
