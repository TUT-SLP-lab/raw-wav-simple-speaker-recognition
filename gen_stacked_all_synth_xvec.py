import os

import hydra
import numpy as np
import torch
from omegaconf import DictConfig


def load_data(path: str) -> torch.Tensor:
    data = np.load(path)
    data_tensor = torch.from_numpy(data)
    return data_tensor


@hydra.main(config_path="conf/simpleSR", config_name="extra_stacked_all_config", version_base="1.3")
def main(config: DictConfig):
    xvector_dir = config.outputs.dir
    stacked_xvector_matrix = torch.zeros((config.num_spks + 1, config.xvector_dim))
    data_dir = os.path.join(xvector_dir, "mid", config.mid_type)
    with open(os.path.join(xvector_dir, "mid", config.list_file_name), "r") as f:
        for line in f.readlines():
            name, id = line.strip().split(":")
            stacked_xvector_matrix[int(id)] = load_data(os.path.join(data_dir, name + ".npy"))
    np.save(f"{xvector_dir}/{config.outputs.stack_file_name}", stacked_xvector_matrix.numpy())
    print(f"save to: {xvector_dir}/{config.outputs.stack_file_name}")
    print(stacked_xvector_matrix.shape)


if __name__ == "__main__":
    main()
