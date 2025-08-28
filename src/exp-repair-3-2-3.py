import os, sys, subprocess
import numpy as np
from itertools import product
from utils.constant import Experiment3, ExperimentRepair1, ExperimentRepair2

NUM_REPS = 5

if __name__ == "__main__":
    ds_list = ["c100", "tiny-imagenet"]
    k_list = [0]
    tgt_rank_list = [1, 2, 3]
    misclf_type_list = ["src_tgt", "tgt"]
    fpfn_list = [None, "fp", "fn"]
    fl_method_list = ["ours", "random"]
    
    alpha = float(10/11)  # same as the original Arachne paper, but the scale is different (we set this value so that the sum of weights becomes 1).
    w_num = 11            # average number of weights identified by Arachne
    beta = None           # Previously tried beta=1, but now using the method without explicitly setting beta
    
    for ds, k, tgt_rank, misclf_type, fpfn, fl_method in product(ds_list, k_list, tgt_rank_list, misclf_type_list, fpfn_list, fl_method_list):
        # For src_tgt or all, fpfn must be None
        if (misclf_type == "src_tgt" or misclf_type == "all") and fpfn is not None:
            continue
        # For tgt, fpfn must be specified
        if misclf_type == "tgt" and fpfn is None:
            continue
        
        # Repeat because the repair search process itself involves randomness
        for reps_id in range(NUM_REPS):
            print(f"{'='*90}\nProcessing: ds={ds}, k={k}, tgt_rank={tgt_rank}, alpha={alpha}, misclf_type={misclf_type}, fpfn={fpfn}, fl_method={fl_method}, reps_id={reps_id}")
            cmd = [
                "python",
                "exp-repair-3-2-2.py",
                ds,
                str(k),
                str(tgt_rank),
                str(reps_id),
                str(w_num),
                "--custom_alpha", str(alpha),
                "--misclf_type", misclf_type,
                "--custom_bounds", "Arachne",
                "--fl_method", fl_method
            ]
            if beta:
                cmd.extend(["--beta", str(beta)])
            # Add fpfn option only if fpfn is not None
            if fpfn:
                cmd.extend(["--fpfn", fpfn])
            print(f"Executing the following cmd: {' '.join(cmd)}\n{'='*90}")
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print("Error occurred, exiting.")
                exit(1)
