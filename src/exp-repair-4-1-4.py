import os, sys, subprocess
import numpy as np
from itertools import product
from utils.constant import Experiment3, ExperimentRepair1, ExperimentRepair2
NUM_REPS = 5

if __name__ == "__main__":
    ds = "c100"
    k_list = [0]
    tgt_rank_list = [1, 2, 3]
    # misclf_type_list = ["all", "src_tgt", "tgt"]
    misclf_type_list = ["src_tgt", "tgt"] # allはいらない説ある
    fpfn_list = [None, "fp", "fn"]
    w_num_list = [236, 472, 944] # exp-repair-4.md 参照
    fl_method_list = ["ours", "bl", "random"]
    alpha = float(10/11) # same as the original Arachne paper, but the scale is different (we set this value so that the sum of weight becomes 1).
    
    for k, tgt_rank, misclf_type, fpfn, w_num, fl_method in product(k_list, tgt_rank_list, misclf_type_list, fpfn_list, w_num_list, fl_method_list):
        if (misclf_type == "src_tgt" or misclf_type == "all") and fpfn is not None: # misclf_type == "src_tgt" or "all"の時はfpfnはNoneだけでいい
            # print(f"Skipping: misclf_type={misclf_type} with fpfn={fpfn} is not valid.")
            continue
        if misclf_type == "tgt" and fpfn is None:
            # print(f"Skipping: misclf_type={misclf_type} with fpfn={fpfn} is not valid.")
            continue
        # repair search自体にランダム性があるので繰り返し
        # for reps_id in range(NUM_REPS):
        for reps_id in range(NUM_REPS):
            print(f"{'='*90}\nProcessing: ds={ds}, k={k}, tgt_rank={tgt_rank}, alpha={alpha}, misclf_type={misclf_type}, fpfn={fpfn}, fl_method={fl_method}, reps_id={reps_id}")
            cmd = [
                "python", 
                "exp-repair-4-1-3.py", 
                "c100",
                str(k),
                str(tgt_rank),
                str(reps_id),
                str(w_num),
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
