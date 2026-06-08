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

# T_select Rep values from the paper (Table: RQ2)
PAPER_T_SELECT_REP = {
    ("c100",         1, "src_tgt", None): 21.06,
    ("c100",         1, "tgt",     "fp"): 11.73,
    ("c100",         1, "tgt",     "fn"): 11.66,
    ("c100",         2, "src_tgt", None): 11.68,
    ("c100",         2, "tgt",     "fp"): 11.78,
    ("c100",         2, "tgt",     "fn"): 11.68,
    ("c100",         3, "src_tgt", None): 11.76,
    ("c100",         3, "tgt",     "fp"): 11.73,
    ("c100",         3, "tgt",     "fn"): 11.74,
    ("tiny-imagenet", 1, "src_tgt", None): 32.73,
    ("tiny-imagenet", 1, "tgt",     "fp"): 21.60,
    ("tiny-imagenet", 1, "tgt",     "fn"): 21.14,
    ("tiny-imagenet", 2, "src_tgt", None): 21.01,
    ("tiny-imagenet", 2, "tgt",     "fp"): 21.30,
    ("tiny-imagenet", 2, "tgt",     "fn"): 21.18,
    ("tiny-imagenet", 3, "src_tgt", None): 21.08,
    ("tiny-imagenet", 3, "tgt",     "fp"): 21.17,
    ("tiny-imagenet", 3, "tgt",     "fn"): 21.09,
}


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

        paper_rep = PAPER_T_SELECT_REP.get((ds, tgt_rank, misclf_type, fpfn))
        ig_time = d.get("ig_time_sec", d.get("ig_mean_sec"))
        records.append({
            "ds":           ds,
            "k":            k,
            "tgt_rank":     tgt_rank,
            "misclf_type":  misclf_type,
            "fpfn":         fpfn,
            "ig_time_sec":        ig_time,
            "paper_t_select_rep": paper_rep,
            "speedup_vs_paper":   ig_time / paper_rep if paper_rep else None,
        })
    return pd.DataFrame(records)


if __name__ == "__main__":
    df = load_results()
    if df.empty:
        print("[INFO] No results found. Run exp-ig-1-2.py first.")
        exit(0)

    print(f"\nLoaded {len(df)} results.\n")

    # ── Per-pair comparison table ─────────────────────────────────────────────
    print("=" * 80)
    print(f"{'ds':<16} {'rank':>4} {'type':<12} {'IG(sec)':>10} {'Rep paper':>10} {'speedup':>8}")
    print("=" * 80)
    for _, row in df.iterrows():
        fpfn_str = row["fpfn"] if row["fpfn"] else "-"
        type_str = f"{row['misclf_type']}/{fpfn_str}"
        speedup_str = f"{row['speedup_vs_paper']:.1f}x" if row["speedup_vs_paper"] else "N/A"
        print(f"{row['ds']:<16} {row['tgt_rank']:>4} {type_str:<12} "
              f"{row['ig_time_sec']:>10.2f} {row['paper_t_select_rep']:>10.2f} {speedup_str:>8}")

    # ── Per-dataset summary ───────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("Per-dataset summary")
    print("=" * 80)
    for ds in df["ds"].unique():
        sub = df[df["ds"] == ds]
        print(f"\n[{ds}]")
        print(f"  IG          : {sub['ig_time_sec'].mean():.2f} sec (mean)")
        print(f"  Rep (paper) : {sub['paper_t_select_rep'].mean():.2f} sec (mean)")
        print(f"  Speedup     : {sub['speedup_vs_paper'].mean():.1f}x  (median {sub['speedup_vs_paper'].median():.1f}x)")

    # ── Overall summary ───────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("Overall summary (all datasets & conditions)")
    print("=" * 80)
    print(f"  IG          : {df['ig_time_sec'].mean():.2f} sec  (median {df['ig_time_sec'].median():.2f})")
    print(f"  Rep (paper) : {df['paper_t_select_rep'].mean():.2f} sec  (median {df['paper_t_select_rep'].median():.2f})")
    print(f"  Speedup     : {df['speedup_vs_paper'].mean():.1f}x  "
          f"(median {df['speedup_vs_paper'].median():.1f}x, "
          f"min {df['speedup_vs_paper'].min():.1f}x, max {df['speedup_vs_paper'].max():.1f}x)")

    # ── Save CSV ─────────────────────────────────────────────────────────────
    out_csv = "exp-ig-1-3_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[INFO] Full results saved to {out_csv}")

