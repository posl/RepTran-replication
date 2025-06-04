import os, sys, subprocess
import numpy as np
from itertools import product
from utils.constant import Experiment3, ExperimentRepair1, ExperimentRepair2
NUM_REPS = 5

if __name__ == "__main__":
    ds = "tiny-imagenet"
    # k_list = range(5)
    # tgt_rank_list = range(1, 6)
    
    # TODO: BELOW SHOULD BE CHANGED FOR EACH RUN.
    k_list = [0]
    tgt_rank_list = [1, 2, 3]
    
    # misclf_type_list = ["all", "src_tgt", "tgt"]
    misclf_type_list = ["src_tgt", "tgt"] # allはいらない説ある
    
    fpfn_list = [None, "fp", "fn"]
    alpha = float(10/11)
    w_num = 11 # Arachneで特定された重み数の平均
    beta = None # XXX beta = 1を指定してたけど指定しない方法にしてみた XXX
    
    # fl_method_list = ["ours"]  # 追加: random methodも試す場合
    fl_method_list = ["ours", "random"]  # 追加: random methodも試す場合
    
    tgt_split_list = ["repair", "test"]
    
    for k, tgt_rank, misclf_type, fpfn, fl_method, tgt_split in product(
        k_list, tgt_rank_list,  misclf_type_list, fpfn_list, fl_method_list, tgt_split_list
    ):
        if (misclf_type == "src_tgt" or misclf_type == "all") and fpfn is not None: # misclf_type == "src_tgt" or "all"の時はfpfnはNoneだけでいい
            continue
        if misclf_type == "tgt" and fpfn is None:
            continue
        # repair search自体にランダム性があるので繰り返し
        for reps_id in range(NUM_REPS):
            print(f"{'='*90}\nProcessing: ds={ds}, k={k}, tgt_rank={tgt_rank}, alpha={alpha}, misclf_type={misclf_type}, fpfn={fpfn}, fl_method={fl_method}, reps_id={reps_id}")
            cmd = [
                "python", 
                "exp-repair-3-2-4.py", 
                ds,
                str(k),
                str(tgt_rank),
                str(reps_id),
                str(w_num),
                "--custom_alpha", str(alpha), 
                "--misclf_type", misclf_type, 
                "--custom_bounds", "Arachne", 
                "--fl_method", fl_method,
                "--tgt_split", tgt_split
            ]
            if beta:
                cmd.extend(["--beta", str(beta)])
            if fpfn:  # fpfnがNoneでない場合のみ追加
                cmd.extend(["--fpfn", fpfn])
            print(f"Executing the following cmd: {' '.join(cmd)}\n{'='*90}")
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print("Error occurred, exiting.")
                exit(1)
