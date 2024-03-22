import copy
import datetime
import os

import numpy as np
import torch
from torch.nn.functional import cosine_similarity

# memo
# 1. オリジナルファイルを読み出す
#     - 連想配列にspk_idをキーとして保存
# 2. reconstruction_xvectorを読み出す
#     - 連想配列にspk_id


def load_xvec(path):
    """Load x-vectors from a file.

    Args:
        path (str): Path to the x-vector file.

    Returns:
        torch.Tensor: A tensor of size (num_utts, num_dims).
    """
    return torch.from_numpy(np.load(path)).unsqueeze(0)


if __name__ == "__main__":

    spk_list_path = "data/reconstruction/spk.list"
    utt_list_path = "data/reconstruction/eval.list"
    utt_data_path = "exp/simpleSR/2023-12-06_16:25:27/output"

    # spk_list_path = "data/reconstruction/mid_gen_all/spk.list"
    # utt_list_path = "data/reconstruction/mid_gen_all/eval.list"
    # utt_data_path = "exp/simpleSR/2023-12-10_20:22:08_norm_64/output"

    with open(spk_list_path) as f:
        spk_dict = {
            spk.split("/")[-1]: load_xvec(
                os.path.join("outputs/x_vec_mean", f"{ spk + ( '_mean' if 'mid' not in spk else '' ) }.npy")
            )
            for spk in f.read().splitlines()
        }

    with open(utt_list_path) as f:
        utt_list = f.read().splitlines()

    output_dir_path = f"outputs/reconstruction/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(output_dir_path, exist_ok=True)
    worth_result = ["utt_id,spk_id,cosine_similarity"]

    cos_sim_list = {spk: [] for spk in spk_dict.keys()}

    # Compute cosine similarity.
    with open(os.path.join(output_dir_path, "result.csv"), "w") as f:
        f.write("utt_id,spk_id,cosine_similarity\n")
        for utt in utt_list:
            tmp_spk = utt.split("_")[0:2]
            spk = "_".join(tmp_spk) if "jvs" in tmp_spk[1] else tmp_spk[0]
            cos_sim = float(cosine_similarity(load_xvec(os.path.join(utt_data_path, f"{utt}.npy")), spk_dict[spk]))
            cos_sim_list[spk].append(cos_sim)
            if cos_sim < 0.8:
                worth_result.append(f"{utt},{spk},{cos_sim}")
            f.write(f"{utt},{spk},{cos_sim}\n")
    with open(os.path.join(output_dir_path, "worth_result.csv"), "w") as f:
        f.write("\n".join(worth_result))

    # Compute mean and std.
    single_spks = set()
    for d_spk in spk_dict.keys():
        tmp = d_spk.split("_")
        if len(tmp) == 1:
            single_spks.add(d_spk)
        else:
            left, right = tmp
            single_spks.add(spk)
            single_spks.add(right)
    single_spks = sorted(list(single_spks))
    mean_matrix = {spk: {_spk: None for _spk in single_spks} for spk in single_spks}
    std_matrix = {spk: {_spk: None for _spk in single_spks} for spk in single_spks}
    with open(os.path.join(output_dir_path, "mean_std.csv"), "w") as f:
        f.write("spk_id,mean,std\n")
        for spk, cos_sim in cos_sim_list.items():
            if len(cos_sim) == 0:
                continue
            mean = np.mean(cos_sim)
            std = np.std(cos_sim)
            tmp = spk.split("_")
            if len(tmp) == 1:
                mean_matrix[spk][spk] = mean
                std_matrix[spk][spk] = std
            else:
                left, right = tmp
                mean_matrix[left][right] = mean
                mean_matrix[right][left] = mean
                std_matrix[left][right] = std
                std_matrix[right][left] = std
            f.write(f"{spk},{mean},{std}\n")

    # save matrix
    with open(os.path.join(output_dir_path, "mean_matrix.csv"), "w") as mean_f, open(
        os.path.join(output_dir_path, "std_matrix.csv"), "w"
    ) as std_f:
        first_line = "spk_id," + ",".join(single_spks) + "\n"
        mean_f.write(first_line)
        std_f.write(first_line)
        for left in single_spks:
            _tmp_mean = [str(mean_matrix[left][right]) for right in single_spks]
            _tmp_std = [str(std_matrix[left][right]) for right in single_spks]
            mean_f.write(left + "," + ",".join(_tmp_mean) + "\n")
            std_f.write(left + "," + ",".join(_tmp_std) + "\n")
