"""
exp-repair-7-1-2.py  —  launcher for exp-repair-7-1-1.py (Meta3 / RQ6)

Iterates over all 9 misclassification benchmarks x 5 reps x 2 new score_modes x 2 datasets
= 180 repair runs (the Full=ours and No-neuron-score=bl conditions are reused from
exp-repair-4-1 and are NOT launched here).

Per-benchmark incremental save + resume (skips if the patch already exists),
subprocess with retry, USE_TF=0.
"""
import os
import subprocess
from itertools import product

import sys
sys.path.insert(0, os.path.dirname(__file__))
from utils.constant import ViTExperiment

NUM_REPS     = 5
FIXED_WNUM   = 472
FIXED_ALPHA  = 10 / (1 + 10)
FIXED_BOUNDS = "Arachne"
MAX_RETRY    = 3


def get_patch_path(ds, k, tgt_rank, misclf_type, fpfn, score_mode, reps_id):
    pretrained_dir = getattr(ViTExperiment, ds.replace("-", "_")).OUTPUT_DIR.format(k=k)
    misclf_ptn = f"{misclf_type}_{fpfn}" if (fpfn is not None and misclf_type == "tgt") else misclf_type
    save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_ptn}_repair_weight_by_de")
    setting_id = f"n{FIXED_WNUM}_alpha{FIXED_ALPHA}_bounds{FIXED_BOUNDS}_{score_mode}_ours"
    return os.path.join(save_dir, f"exp-repair-7-1-best_patch_{setting_id}_reps{reps_id}.npy")


if __name__ == "__main__":
    ds_list          = ["c100", "tiny-imagenet"]
    k_list           = [0]
    tgt_rank_list    = [1, 2, 3]
    misclf_type_list = ["src_tgt", "tgt"]
    fpfn_list        = [None, "fp", "fn"]
    score_mode_list  = ["vdiff", "misact"]

    env = os.environ.copy()
    env["USE_TF"] = "0"

    for ds, k, tgt_rank, misclf_type, fpfn, score_mode in product(
        ds_list, k_list, tgt_rank_list, misclf_type_list, fpfn_list, score_mode_list
    ):
        if (misclf_type == "src_tgt") and fpfn is not None:
            continue
        if misclf_type == "tgt" and fpfn is None:
            continue

        for reps_id in range(NUM_REPS):
            patch_path = get_patch_path(ds, k, tgt_rank, misclf_type, fpfn, score_mode, reps_id)
            if os.path.exists(patch_path):
                print(f"[SKIP] already exists: {patch_path}")
                continue
            print(f"{'='*90}\nProcessing: ds={ds}, k={k}, tgt_rank={tgt_rank}, "
                  f"misclf_type={misclf_type}, fpfn={fpfn}, score_mode={score_mode}, reps_id={reps_id}")
            cmd = [
                "python", "exp-repair-7-1-1.py",
                ds, str(k), str(tgt_rank), str(reps_id),
                "--score_mode", score_mode,
                "--misclf_type", misclf_type,
            ]
            if fpfn is not None:
                cmd.extend(["--fpfn", fpfn])

            print(f"Executing: {' '.join(cmd)}\n{'='*90}")
            for attempt in range(1, MAX_RETRY + 1):
                result = subprocess.run(cmd, env=env)
                if result.returncode == 0 and os.path.exists(patch_path):
                    break
                print(f"[RETRY {attempt}/{MAX_RETRY}] failed (rc={result.returncode}) for {patch_path}")
            else:
                print(f"[ERROR] giving up after {MAX_RETRY} attempts: {patch_path}")
                exit(1)
