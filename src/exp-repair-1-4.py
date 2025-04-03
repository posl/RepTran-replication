import os, sys, subprocess
import numpy as np
from itertools import product
from utils.constant import Experiment3, ExperimentRepair1, ExperimentRepair2
NUM_REPS = 5

if __name__ == "__main__":
    ds = "c100"
    # k_list = range(5)
    # tgt_rank_list = range(1, 6)
    
    # TODO: BELOW SHOULD BE CHANGED FOR EACH RUN.
    k_list = [0]
    tgt_rank_list = [1, 2, 3]
    
    # misclf_type_list = ["all", "src_tgt", "tgt"]
    misclf_type_list = ["src_tgt", "tgt"] # allはいらない説ある
    
    fpfn_list = [None, "fp", "fn"]
    alpha_list = [0.2, 0.4, 0.6, 0.8]
    
    fl_method_list = ["vmg", "random", "bl"]
    # fl_method_list = ["vmg"] # いったんvmgだけやって時間みたい
    
    # tgt_split_list = ["repair", "test"]
    tgt_split_list = ["repair"] # NOTE: THIS IS HARD CODED FOR NOW.
    
    exp_list = [ExperimentRepair1, ExperimentRepair2]
    
    for k, tgt_rank, misclf_type, fpfn, fl_method, alpha, exp, tgt_split in product(
        k_list, tgt_rank_list,  misclf_type_list, fpfn_list, fl_method_list, alpha_list, exp_list, tgt_split_list
    ):
        if (misclf_type == "src_tgt" or misclf_type == "all") and fpfn is not None: # misclf_type == "src_tgt" or "all"の時はfpfnはNoneだけでいい
            continue
        if misclf_type == "all" and tgt_rank != 1:
            continue
        if fl_method == "vmg":
            n = exp.NUM_IDENTIFIED_NEURONS_RATIO
            wnum = exp.NUM_IDENTIFIED_WEIGHTS
            wnum = 8 * wnum * wnum
        else:
            n = exp.NUM_IDENTIFIED_WEIGHTS
            wnum = None # vmg以外はwnumがファイル名に入ってないので
        # repair search自体にランダム性があるので繰り返し
        for reps_id in range(NUM_REPS):
            cmd = [
                "python", 
                "exp-repair-1-3.py", 
                "c100",
                str(k),
                str(tgt_rank),
                str(reps_id),
                "--custom_n", str(n), 
                "--custom_alpha", str(alpha), 
                "--misclf_type", misclf_type, 
                "--custom_bounds", "Arachne", 
                "--fl_method", fl_method,
                "--tgt_split", tgt_split
            ]
            if fpfn:  # fpfnがNoneでない場合のみ追加
                cmd.extend(["--fpfn", fpfn])
            if wnum:
                cmd.extend(["--custom_wnum", str(wnum)])
            print(f"Executing the following cmd: {' '.join(cmd)}\n{'='*90}")
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print("Error occurred, exiting.")
                exit(1)
