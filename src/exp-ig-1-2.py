"""
exp-ig-1-2.py  —  launcher for exp-ig-1-1.py

Runs overhead comparison across datasets, tgt_ranks, and misclf_types.
"""
import subprocess
from itertools import product

if __name__ == "__main__":
    ds_list          = ["c100", "tiny-imagenet"]
    k_list           = [0]
    tgt_rank_list    = [1, 2, 3]
    misclf_type_list = ["src_tgt", "tgt"]
    fpfn_list        = [None, "fp", "fn"]
    ig_start_layer   = 11   # last layer only (conservative comparison)
    n_reps           = 1

    for ds, k, tgt_rank, misclf_type, fpfn in product(
        ds_list, k_list, tgt_rank_list, misclf_type_list, fpfn_list
    ):
        if misclf_type == "src_tgt" and fpfn is not None:
            continue
        if misclf_type == "tgt" and fpfn is None:
            continue

        cmd = [
            "python", "exp-ig-1-1.py",
            ds, str(k), str(tgt_rank), misclf_type,
            "--n_reps", str(n_reps),
            "--ig_start_layer", str(ig_start_layer),
        ]
        if fpfn is not None:
            cmd.extend(["--fpfn", fpfn])

        print(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("Error occurred, exiting.")
            exit(1)
