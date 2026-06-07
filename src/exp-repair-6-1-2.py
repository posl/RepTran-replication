"""
exp-repair-6-1-2.py  —  launcher for exp-repair-6-1-1.py

Iterates over all benchmark conditions and p values.
"""
import os
import subprocess
from itertools import product

import sys
sys.path.insert(0, os.path.dirname(__file__))
from utils.constant import ViTExperiment

NUM_REPS     = 5
FIXED_WNUM   = 236
FIXED_ALPHA  = 10 / (1 + 10)
FIXED_BOUNDS = "Arachne"


def get_patch_path(ds, k, tgt_rank, misclf_type, fpfn, p, reps_id):
    pretrained_dir = getattr(ViTExperiment, ds.replace("-", "_")).OUTPUT_DIR.format(k=k)
    misclf_ptn = f"{misclf_type}_{fpfn}" if (fpfn is not None and misclf_type == "tgt") else misclf_type
    save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_ptn}_repair_weight_by_de")
    setting_id = f"n{FIXED_WNUM}_alpha{FIXED_ALPHA}_bounds{FIXED_BOUNDS}_p{p}_ours"
    return os.path.join(save_dir, f"exp-repair-6-1-best_patch_{setting_id}_reps{reps_id}.npy")

if __name__ == "__main__":
    ds_list          = ["c100", "tiny-imagenet"]
    k_list           = [0]
    tgt_rank_list    = [1, 2, 3]
    misclf_type_list = ["src_tgt", "tgt"]
    fpfn_list        = [None, "fp", "fn"]
    p_list           = [0.1, 0.9]

    for ds, k, tgt_rank, misclf_type, fpfn, p in product(
        ds_list, k_list, tgt_rank_list, misclf_type_list, fpfn_list, p_list
    ):
        if (misclf_type == "src_tgt") and fpfn is not None:
            continue
        if misclf_type == "tgt" and fpfn is None:
            continue

        for reps_id in range(NUM_REPS):
            patch_path = get_patch_path(ds, k, tgt_rank, misclf_type, fpfn, p, reps_id)
            # if os.path.exists(patch_path):
            #     print(f"[SKIP] already exists: {patch_path}")
            #     continue
            print(f"{'='*90}\nProcessing: ds={ds}, k={k}, tgt_rank={tgt_rank}, "
                  f"misclf_type={misclf_type}, fpfn={fpfn}, p={p}, reps_id={reps_id}")
            cmd = [
                "python", "exp-repair-6-1-1.py",
                ds, str(k), str(tgt_rank), str(reps_id),
                "--p", str(p),
                "--misclf_type", misclf_type,
            ]
            if fpfn is not None:
                cmd.extend(["--fpfn", fpfn])

            print(f"Executing: {' '.join(cmd)}\n{'='*90}")
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print("Error occurred, exiting.")
                exit(1)
