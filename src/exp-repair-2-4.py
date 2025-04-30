#!/usr/bin/env python
# exp-repair-2-4.py  ← ファイル名は自由

import subprocess
from itertools import product

DS_NAME = "c100"

if __name__ == "__main__":
    k_list           = [0]
    tgt_rank_list    = [1, 2, 3]
    num_reps         = 5
    misclf_type_list = ["src_tgt", "tgt"]
    fpfn_list        = [None, "fp", "fn"]
    alpha_list       = [0.2, 0.4, 0.6, 0.8]
    r                = 16
    tgt_splits       = ["repair", "test"]   # 🆕 どちらも集計したい場合

    # 最初に実験設計を表示
    print("=" * 80)
    print("Experiment Design:")
    print(f"DS_NAME: {DS_NAME}")
    print(f"k_list: {k_list}")
    print(f"tgt_rank_list: {tgt_rank_list}")
    print(f"num_reps: {num_reps}")
    print(f"misclf_type_list: {misclf_type_list}")
    print(f"fpfn_list: {fpfn_list}")
    print(f"alpha_list: {alpha_list}")
    print(f"r: {r}")
    print("=" * 80)

    for k, tgt_rank, misclf_type, fpfn, alpha, reps_id, split in product(
            k_list, tgt_rank_list, misclf_type_list,
            fpfn_list, alpha_list, range(num_reps), tgt_splits):

        if misclf_type in {"src_tgt", "all"} and fpfn is not None:
            continue
        if misclf_type == "all" and tgt_rank != 1:
            continue

        cmd = [
            "python", "exp-repair-2-3.py",
            DS_NAME,           # ds
            str(k),            # k
            str(tgt_rank),     # tgt_rank
            str(reps_id),      # reps_id  ←★ ここを位置引数4番目に
            "--misclf_type", misclf_type,
            "--alpha", str(alpha),
            "--r",     str(r),
            "--tgt_split", split,
        ]
        if fpfn:
            cmd.extend(["--fpfn", fpfn])

        print("=" * 80)
        print("Executing:", " ".join(cmd))
        print("=" * 80)

        res = subprocess.run(cmd)
        if res.returncode != 0:
            raise SystemExit("✖️ exp-repair-2-3 failed")