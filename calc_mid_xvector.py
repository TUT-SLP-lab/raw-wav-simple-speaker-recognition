import os
from copy import deepcopy
from itertools import combinations

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

# argparseで以下のように引数を設定する
# --typeは特定の配列から選択する必要がある
# python calc_mid_xvector.py --type simple
# python calc_mid_xvector.py --type unit_vector


def load_data(path: str) -> torch.Tensor:
    data = np.load(path)
    data_tensor = torch.from_numpy(data)
    return data_tensor


def simple(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return (x1 + x2) / 2


def calc_normalized(x1: torch.Tensor) -> torch.Tensor:
    return x1 / torch.norm(x1)


def simple_norm(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    x1_norm = calc_normalized(x1)
    x2_norm = calc_normalized(x2)
    tmp = (x1_norm + x2_norm) / 2
    return calc_normalized(tmp)


def outer_harf_norm(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    x1_norm = torch.norm(x1)
    x2_norm = torch.norm(x2)
    x1_unit = x1 / x1_norm
    x2_unit = x2 / x2_norm
    outer_unit = torch.outer(x1_unit, x2_unit)
    return (outer_unit) * (x1_norm + x2_norm) / 2


def outer_unit(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    x1_unit = x1 / torch.norm(x1)
    x2_unit = x2 / torch.norm(x2)
    return torch.outer(x1_unit, x2_unit)


@hydra.main(config_path="conf/simpleSR", config_name="mid_gen_config", version_base="1.3")
def main(config: DictConfig):
    mid_type = config.mid_type
    calc_mid_xvector = None
    if mid_type == "simple":
        calc_mid_xvector = simple
    elif mid_type == "simple_norm":
        calc_mid_xvector = simple_norm
    elif mid_type == "outer_harf_norm":
        calc_mid_xvector = outer_harf_norm
    elif mid_type == "outer_unit":
        calc_mid_xvector = outer_unit
    else:
        raise ValueError(f"mid_type: '{mid_type}' is not defined")

    with open(config.data.spk.female.file) as f:
        female_spks = f.read().splitlines()
    xvector_dir = config.xvector_dir
    output_dir = os.path.join(xvector_dir, "mid", mid_type)
    os.makedirs(output_dir, exist_ok=True)
    memo = {}
    for spk_a, spk_b in combinations(female_spks, 2):
        if spk_a in memo:
            x1 = memo[spk_a]
        else:
            x1 = load_data(f"{xvector_dir}/{spk_a}_mean.npy")
            memo[spk_a] = deepcopy(x1)
        if spk_b in memo:
            x2 = memo[spk_b]
        else:
            x2 = load_data(f"{xvector_dir}/{spk_b}_mean.npy")
            memo[spk_b] = deepcopy(x2)
        x_mid = calc_mid_xvector(x1, x2).numpy()
        np.save(f"{output_dir}/{spk_a}_{spk_b}.npy", x_mid)


if __name__ == "__main__":
    main()
