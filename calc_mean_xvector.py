# 平均をとって保存する

import argparse
import os

import numpy as np
import torch


def load_data(path: str) -> torch.Tensor:
    data = np.load(path)
    data_tensor = torch.from_numpy(data)
    return data_tensor


def calc_mean(datas, isNormalize) -> torch.Tensor:
    tmp = torch.stack(datas).mean(dim=0)
    if isNormalize:
        return torch.div(tmp, torch.norm(tmp))
    return tmp


def get_file_list(path: str) -> dict:
    spk_xvecs = {}
    for file_name in os.listdir(path):
        spk = file_name[0:6]
        if spk not in spk_xvecs:
            spk_xvecs[spk] = []
        spk_xvecs[spk].append(os.path.join(path, file_name))
    return spk_xvecs


def save_mean_xvec(path: str, mean_xvec: torch.Tensor) -> None:
    np.save(path, mean_xvec.numpy())


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="/home/hosoi/git/ssr/exp/simpleSR/2023-10-31_16:21:28/output",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/home/hosoi/git/ssr/outputs/x_vec_mean",
    )
    parser.add_argument("--normalize", action='store_true')
    return parser.parse_args()


def main(path: str, save_path: str) -> None:
    os.makedirs(save_path, exist_ok=True)
    spk_xvecs = get_file_list(path)
    for spk, xvecs in spk_xvecs.items():
        mean_xvec = calc_mean([load_data(xvec) for xvec in xvecs], args.normalize)
        save_mean_xvec(os.path.join(save_path, spk + "_mean.npy"), mean_xvec)


if __name__ == "__main__":
    args = arg_parse()
    main(args.path, args.save_path)
