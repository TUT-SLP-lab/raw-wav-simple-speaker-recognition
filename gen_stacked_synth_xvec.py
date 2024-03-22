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


@hydra.main(config_path="conf/simpleSR", config_name="config", version_base="1.3")
def main(config: DictConfig):
    args = arg_parse()
    xvector_dir = args.xvector_dir
    spk_list = []
    with open("outputs/x_vec_mean/mid/mid_id.list", "r") as f:
        for line in f.readlines():
            name, id = line.strip().split(":")
            spk_list.append((name, int(id)))
    stacked_xvector_matrix = load_data(f"{xvector_dir}/stacked_xvector_matrix.npy")
    for name, id in spk_list:
        stacked_xvector_matrix[id] = load_data(f"{xvector_dir}/mid/simple/{name}.npy")
    np.save(f"{xvector_dir}/stacked_mid_mix_xvector.npy", stacked_xvector_matrix.numpy())


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xvector_dir",
        type=str,
        default="/home/hosoi/git/ssr/outputs/x_vec_mean",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
