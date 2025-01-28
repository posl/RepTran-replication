import os, sys, subprocess
import numpy as np
from itertools import product
from utils.constant import Experiment3

if __name__ == "__main__":
    ds = "c100"
    k_list = range(5)
    tgt_rank_list = range(1, 6)
    misclf_type_list = ["all", "src_tgt", "tgt"]
    fpfn_list = [None, "fp", "fn"]
    alpha_list = [0.2, 0.4, 0.6, 0.8]
    # n_list = get_nlist()
    n_list = [96]
    bounds_list = ["ContrRep", "Arachne"]
    fl_method_list = ["random", "vmg", "bl", "vdiff"]
    
    for k, tgt_rank, n, alpha, misclf_type, fpfn, fl_method in product(
        k_list, tgt_rank_list, n_list, alpha_list, misclf_type_list, fpfn_list, fl_method_list
    ):
        if (misclf_type == "src_tgt" or misclf_type == "all") and fpfn is not None: # misclf_type == "src_tgt" or "all"の時はfpfnはNoneだけでいい
            continue
        if misclf_type == "all" and tgt_rank != 1:
            continue
        print(f"{'='*90}\nProcessing: ds={ds}, k={k}, tgt_rank={tgt_rank}, n={n}, alpha={alpha}, misclf_type={misclf_type}, fpfn={fpfn}, fl_method={fl_method}")
        if fl_method == "vmg":
            n = Experiment3.NUM_IDENTIFIED_NEURONS_RATIO
            wnum = Experiment3.NUM_TOTAL_WEIGHTS
        # repair search自体にランダム性があるので繰り返し
        for reps_id in range(Experiment3.NUM_REPS):
            cmd = [
                "python", 
                "exp-repair-1-1.py", 
                "c100",
                str(k),
                str(tgt_rank),
                str(reps_id),
                "--custom_n", str(n), 
                "--custom_alpha", str(alpha), 
                "--misclf_type", misclf_type, 
                "--custom_bounds", "Arachne", 
                "--fl_method", fl_method
            ]
            if fpfn:  # fpfnがNoneでない場合のみ追加
                cmd.extend(["--fpfn", fpfn])
            print(f"Executing the following cmd: {' '.join(cmd)}\n{'='*90}")
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print("Error occurred, exiting.")
                exit(1)
