import os, sys, subprocess
import numpy as np
from itertools import product

def get_nlist():
    # あるnが対象FFNの何%のニューロン数に対応するかを返す
    get_change_ratio = lambda n: 800 * n * n / 4718592
    nlist, rlist = [], []
    for n in range(5, 769):
        r = get_change_ratio(n)
        if r <= 100:
            nlist.append(n)
            rlist.append(r)
    # 1, 5, 10x% (x=1,..,9)に最も近くなるnを取得
    target_ratios = np.array([1, 2, 3, 4, 5, 10, 15, 20]) # NOTE: hard coded
    # target_ratios = np.array([1]) # NOTE: hard coded
    tgt_nlist = []
    for target in target_ratios:
        closest_n = min(zip(nlist, rlist), key=lambda x: abs(x[1] - target)) # keyはminの計算の際の大きい小さいの基準
        tgt_nlist.append(closest_n[0])
        print(f"target ratio: {target}%, closest n: {closest_n[0]}, ratio: {closest_n[1]}%, num_weights: {8*closest_n[0]*closest_n[0]}")
    return tgt_nlist

if __name__ == "__main__":
    alpha_list = [0.2, 0.4, 0.6, 0.8]
    n_list = get_nlist()
    bounds_list = ["ContrRep", "Arachne"]
    ds = "c100"
    k = 0
    tgt_rank = 1
    misclf_type = "tgt"
    do_localize = False
    fpfn = "fn"

    for n, alpha, bounds in product(n_list, alpha_list, bounds_list):
        print(f"{'='*60}\nn={n}, alpha={alpha}, bounds={bounds}\n{'='*60}")
        if do_localize:
            cmd = ["python", "007d_localize_weights.py", ds, str(k), str(tgt_rank), str(n), "--misclf_type", misclf_type]
            print(f"executing the following cmd: {cmd}")
            result = subprocess.run(cmd)
            if result.returncode != 0:
                exit(1)
        cmd = ["python", "007e_change_ffn_weights.py", "c100", str(k), str(tgt_rank), "--custom_n", str(n), "--custom_alpha", str(alpha), "--misclf_type", misclf_type, "--custom_bounds", bounds, "--fpfn", fpfn]
        print(f"executing the following cmd: {cmd}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            exit(1)