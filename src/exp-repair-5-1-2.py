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
    w_num_list = [236]  # Reference: exp-repair-5.md
    fl_method_list = ["ours", "bl", "random"]
    
    # Convert alpha from Arachne's notation (alpha_in_arachne) to ours.
    # Formula: alpha = alpha_in_arachne / (1 + alpha_in_arachne)
    alpha_in_arachne_list = [1, 4, 8]
    alpha_list = [float(alpha / (1 + alpha)) for alpha in alpha_in_arachne_list]

    for ds, k, tgt_rank, misclf_type, fpfn, w_num, fl_method, alpha in product(
        ds_list, k_list, tgt_rank_list, misclf_type_list, fpfn_list, w_num_list, fl_method_list, alpha_list
    ):
        # For src_tgt or all, fpfn must be None
        if (misclf_type in ["src_tgt", "all"]) and fpfn is not None:
            continue
        # For tgt, fpfn must be specified
        if misclf_type == "tgt" and fpfn is None:
            continue

        # Repeat because the repair search process itself involves randomness
        for reps_id in range(NUM_REPS):
            print(f"{'='*90}\nProcessing: ds={ds}, k={k}, tgt_rank={tgt_rank}, alpha={alpha}, misclf_type={misclf_type}, fpfn={fpfn}, fl_method={fl_method}, reps_id={reps_id}")
            cmd = [
                "python",
                "exp-repair-5-1-1.py",
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
            # Add fpfn option only if it is not None
            if fpfn:
                cmd.extend(["--fpfn", fpfn])

            print(f"Executing the following cmd: {' '.join(cmd)}\n{'='*90}")
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print("Error occurred, exiting.")
                exit(1)
