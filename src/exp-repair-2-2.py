# run_repair_jobs.py
import os, sys, subprocess
from itertools import product

DS_NAME  = "c100"

if __name__ == "__main__":
    # 変更が必要な場合はここだけ編集
    k_list           = [0]
    tgt_rank_list    = [1, 2, 3]
    num_reps         = 5
    misclf_type_list = ["src_tgt", "tgt"]   # "all" を使わないなら外す
    fpfn_list        = [None, "fp", "fn"]   # src_tgt/all には None だけ使う
    alpha_list = [0.2, 0.4, 0.6, 0.8]
    r = 16
    lora_epoch = 20
    
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
    print(f"lora_epoch: {lora_epoch}")
    print("=" * 80)

    for k, tgt_rank, misclf_type, fpfn, alpha in product(
        k_list, tgt_rank_list, misclf_type_list, fpfn_list, alpha_list
    ):
        # 不要な組合せをスキップ
        if misclf_type in {"src_tgt", "all"} and fpfn is not None:
            continue
        if misclf_type == "all" and tgt_rank != 1:
            continue

        for reps_id in range(num_reps):
            cmd = [
                "python",
                "exp-repair-2-1.py",   # ← 呼び出す先のスクリプト名
                DS_NAME,
                str(k),
                str(tgt_rank),
                str(reps_id),
                "--misclf_type", misclf_type,
                "--r", str(r),
                "--lora_epoch", str(lora_epoch),
                "--alpha", str(alpha),
            ]
            if fpfn:                        # fp / fn のときだけ付ける
                cmd.extend(["--fpfn", fpfn])

            print("=" * 80)
            print("Executing:", " ".join(cmd))
            print("=" * 80)

            result = subprocess.run(cmd)
            if result.returncode != 0:
                print("Error occurred, exiting.")
                sys.exit(1)