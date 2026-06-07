"""
exp-repair-6-1-2.py  —  launcher for exp-repair-6-1-1.py

Iterates over all benchmark conditions and p values.
"""
import subprocess
from itertools import product

NUM_REPS = 1

if __name__ == "__main__":
    ds_list          = ["c100", "tiny-imagenet"]
    k_list           = [0]
    tgt_rank_list    = [1, 2, 3]
    misclf_type_list = ["src_tgt", "tgt"]
    fpfn_list        = [None, "fp", "fn"]
    fl_method_list   = ["ours", "bl", "random"]
    p_list           = [0.1, 0.9]

    for ds, k, tgt_rank, misclf_type, fpfn, fl_method, p in product(
        ds_list, k_list, tgt_rank_list, misclf_type_list, fpfn_list, fl_method_list, p_list
    ):
        if (misclf_type == "src_tgt") and fpfn is not None:
            continue
        if misclf_type == "tgt" and fpfn is None:
            continue

        for reps_id in range(NUM_REPS):
            print(f"{'='*90}\nProcessing: ds={ds}, k={k}, tgt_rank={tgt_rank}, "
                  f"misclf_type={misclf_type}, fpfn={fpfn}, fl_method={fl_method}, "
                  f"p={p}, reps_id={reps_id}")
            cmd = [
                "python", "exp-repair-6-1-1.py",
                ds, str(k), str(tgt_rank), str(reps_id),
                "--p", str(p),
                "--fl_method", fl_method,
                "--misclf_type", misclf_type,
            ]
            if fpfn is not None:
                cmd.extend(["--fpfn", fpfn])

            print(f"Executing: {' '.join(cmd)}\n{'='*90}")
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print("Error occurred, exiting.")
                exit(1)
