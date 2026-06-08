"""
exp-ig-1-3.py  —  aggregate overhead results and print summary table

Loads all JSON files produced by exp-ig-1-1.py and reports:
  - mean IG time, mean REPTRAN time, speedup per dataset
  - overall mean/median speedup
"""
import os, json
import numpy as np
import pandas as pd
from itertools import product
from utils.constant import ViTExperiment

IG_LAYER_STR = "l11tol11"   # matches --ig_start_layer 11 in exp-ig-1-1.py


def load_results():
    ds_list          = ["c100", "tiny-imagenet"]
    k_list           = [0]
    tgt_rank_list    = [1, 2, 3]
    misclf_type_list = ["src_tgt", "tgt"]
    fpfn_list        = [None, "fp", "fn"]

    records = []
    for ds, k, tgt_rank, misclf_type, fpfn in product(
        ds_list, k_list, tgt_rank_list, misclf_type_list, fpfn_list
    ):
        if misclf_type == "src_tgt" and fpfn is not None:
            continue
        if misclf_type == "tgt" and fpfn is None:
            continue

        misclf_ptn = misclf_type if fpfn is None else f"{misclf_type}_{fpfn}"
        pretrained_dir = getattr(ViTExperiment, ds.replace("-", "_")).OUTPUT_DIR.format(k=k)
        json_path = os.path.join(
            pretrained_dir,
            f"misclf_top{tgt_rank}",
            f"exp-ig-1-1_{ds}_k{k}_top{tgt_rank}_{misclf_ptn}_{IG_LAYER_STR}.json"
        )
        if not os.path.exists(json_path):
            print(f"[MISSING] {json_path}")
            continue

        with open(json_path) as f:
            d = json.load(f)

        records.append({
            "ds":           ds,
            "k":            k,
            "tgt_rank":     tgt_rank,
            "misclf_type":  misclf_type,
            "fpfn":         fpfn,
            "n_tgt_mis":    d["n_tgt_mis"],
            "n_correct":    d["n_correct"],
            "ig_time_sec":      d["ig_time_sec"],
            "reptran_time_sec": d["reptran_time_sec"],
            "speedup_x":        d["speedup_x"],
        })
    return pd.DataFrame(records)


if __name__ == "__main__":
    df = load_results()
    if df.empty:
        print("[INFO] No results found. Run exp-ig-1-2.py first.")
        exit(0)

    print(f"\nLoaded {len(df)} results.\n")

    # ── Per-dataset summary ───────────────────────────────────────────────────
    print("=" * 60)
    print("Per-dataset summary")
    print("=" * 60)
    for ds in df["ds"].unique():
        sub = df[df["ds"] == ds]
        print(f"\n[{ds}]")
        print(f"  IG      : {sub['ig_time_sec'].mean():.2f} sec (mean across conditions)")
        print(f"  REPTRAN : {sub['reptran_time_sec'].mean():.2f} sec")
        print(f"  Speedup : {sub['speedup_x'].mean():.1f}x  (median {sub['speedup_x'].median():.1f}x)")

    # ── Overall summary ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Overall summary (all datasets & conditions)")
    print("=" * 60)
    print(f"  IG      : {df['ig_time_sec'].mean():.2f} sec  (median {df['ig_time_sec'].median():.2f})")
    print(f"  REPTRAN : {df['reptran_time_sec'].mean():.2f} sec  (median {df['reptran_time_sec'].median():.2f})")
    print(f"  Speedup : {df['speedup_x'].mean():.1f}x  (median {df['speedup_x'].median():.1f}x, "
          f"min {df['speedup_x'].min():.1f}x, max {df['speedup_x'].max():.1f}x)")

    # ── Save CSV ─────────────────────────────────────────────────────────────
    out_csv = "exp-ig-1-3_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[INFO] Full results saved to {out_csv}")
