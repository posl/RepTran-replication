"""
exp-repair-7-1-4.py  —  launcher for exp-repair-7-1-3.py (evaluation, Meta3 / RQ6)
"""
import os
import subprocess
from itertools import product

NUM_REPS = 5

if __name__ == "__main__":
    ds_list          = ["c100", "tiny-imagenet"]
    k_list           = [0]
    tgt_rank_list    = [1, 2, 3]
    misclf_type_list = ["src_tgt", "tgt"]
    fpfn_list        = [None, "fp", "fn"]
    score_mode_list  = ["vdiff", "misact"]
    tgt_split_list   = ["test"]

    env = os.environ.copy()
    env["USE_TF"] = "0"

    for ds, k, tgt_rank, misclf_type, fpfn, score_mode, tgt_split in product(
        ds_list, k_list, tgt_rank_list, misclf_type_list, fpfn_list, score_mode_list, tgt_split_list
    ):
        if (misclf_type == "src_tgt") and fpfn is not None:
            continue
        if misclf_type == "tgt" and fpfn is None:
            continue

        for reps_id in range(NUM_REPS):
            print(f"{'='*90}\nProcessing: ds={ds}, k={k}, tgt_rank={tgt_rank}, "
                  f"misclf_type={misclf_type}, fpfn={fpfn}, "
                  f"score_mode={score_mode}, tgt_split={tgt_split}, reps_id={reps_id}")
            cmd = [
                "python", "exp-repair-7-1-3.py",
                ds, str(k), str(tgt_rank), str(reps_id),
                "--score_mode", score_mode,
                "--misclf_type", misclf_type,
                "--tgt_split", tgt_split,
            ]
            if fpfn is not None:
                cmd.extend(["--fpfn", fpfn])

            print(f"Executing: {' '.join(cmd)}\n{'='*90}")
            result = subprocess.run(cmd, env=env)
            if result.returncode != 0:
                print("Error occurred, exiting.")
                exit(1)
