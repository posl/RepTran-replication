#!/usr/bin/env python
# rq2_stats_table_rr_br.py  –  RQ-2: weight-variant experiments (RR / BR)

import json, math, itertools, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

# ──────────── Experimental configuration ───────────────────────
DATASETS     = ["c100", "tiny-imagenet"]   # datasets
SPLITS       = ["repair", "test"]          # evaluation splits
K            = 0                           # fold id
TGT_RANKS    = [1, 2, 3]                   # misclassification rank
MISCLF_TPS   = ["src_tgt", "tgt_fp", "tgt_fn"]
WN_LIST      = [11, 236, 472, 944]         # numbers of weights (3 variants)
REPS         = range(5)                    # 5 repetitions
ALPHA_STR    = 10/11
ROOT_TMPL    = "/src/src/out_vit_{ds}_fold{K}"

# Three methods and the keys used in JSON filenames
METHODS = {"ours": "ours", "arachne": "bl", "random": "random"}
PAIRS   = [("ours", "arachne"),
           ("ours", "random"),
           ("arachne", "random")]

METRIC_INFO = dict(
    RR=("repair_rate_tgt",    "Repair Rate"),
    BR=("break_rate_overall", "Break  Rate"),
)

# ──────────── JSON ⇒ value ─────────────────────────────────────
def metric_value(ds, split, mtype, rank, rep,
                 wnum, meth_key, json_key):
    """Open one JSON file and return the requested metric value."""
    base = Path(ROOT_TMPL.format(ds=ds, K=K))
    jdir = base / f"misclf_top{rank}" / f"{mtype}_repair_weight_by_de"
    if wnum != 11:
        fn = (f"exp-repair-4-1-metrics_for_{split}_"
              f"n{wnum}_alpha{ALPHA_STR}_boundsArachne_{meth_key}_reps{rep}.json")
    else:
        # Note: for n=11, filenames differ slightly across methods
        if meth_key == "bl":
            fn = f"exp-repair-3-1-metrics_for_{split}_alpha{ALPHA_STR}_boundsArachne_{meth_key}_reps{rep}.json"
        elif meth_key == "ours" or meth_key == "random":
            fn = f"exp-repair-3-2-metrics_for_{split}_alpha{ALPHA_STR}_boundsArachne_{meth_key}_reps{rep}.json"
        else:
            raise NotImplementedError
    
    with open(jdir / fn) as f:
        return json.load(f)[json_key]

# ──────────── Wilcoxon & Cliff’s Δ (paired) ────────────────────
def paired_cliffs_delta(v1: np.ndarray, v2: np.ndarray) -> float:
    """Paired Cliff’s delta: (n_pos - n_neg) / N using per-pair differences."""
    diff = v1 - v2
    return (np.sum(diff > 0) - np.sum(diff < 0)) / diff.size

def wilcoxon_block(values: dict, show_flag: bool) -> dict:
    """
    values = {method: np.array(15)}
    → Holm-adjusted p-values & signed Cliff’s Δ for each pair.
    """
    out, p_raw = {}, []
    for m1, m2 in PAIRS:
        v1, v2 = values[m1], values[m2]
        if show_flag:
            print(m1, m2)
            print(v1, v2)
            print(paired_cliffs_delta(v1, v2))
        # p-value (return 1.0 if all pairs are equal)
        if np.allclose(v1, v2):
            p = 1.0
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                p = wilcoxon(v1, v2, zero_method="wilcox").pvalue
        d = paired_cliffs_delta(v1, v2)
        tag = f"{m1[:1].upper()}v{m2[:1].upper()}"
        out[f"{tag}_p_raw"] = p
        out[f"{tag}_d"]     = d
        p_raw.append(p)
    # Holm correction across the three pairs
    _, p_adj, _, _ = multipletests(p_raw, method="holm")
    for (m1, m2), p_c in zip(PAIRS, p_adj):
        out[f"{m1[:1].upper()}v{m2[:1].upper()}_p"] = p_c
    return out

def stars(p):
    return "***" if p <= .001 else "**" if p <= .01 else "*" if p <= .05 else ""

def cell(d, p):
    """Format one cell with signed delta and significance stars."""
    return f"'{d:+.2f} {stars(p)}"

# ──────────── main ─────────────────────────────────────────────
per_metric_tables = {}  # Keep per-metric tables for a later merge
show_flag = False
for met_tag, (json_key, nice_name) in METRIC_INFO.items():
    rows = []

    for ds, split, wnum, mtype in itertools.product(
            DATASETS, SPLITS, WN_LIST, MISCLF_TPS):

        # Collect 15 points × 3 methods (3 ranks × 5 reps)
        vals = {m: [] for m in METHODS}
        for rank, rep in itertools.product(TGT_RANKS, REPS):
            for m, key in METHODS.items():
                vals[m].append(metric_value(ds, split, mtype,
                                            rank, rep, wnum,
                                            key, json_key))
        vals = {m: np.asarray(v) for m, v in vals.items()}

        # (Debug toggle for a specific slice)
        if ds == "c100" and wnum == 236 and mtype == "src_tgt" and split == "test":
            print(ds, wnum, mtype)
            show_flag = True
        else:
            show_flag = False

        stat = wilcoxon_block(vals, show_flag)
        rows.append(dict(
            dataset      = ds,
            split        = split,
            wnum         = wnum,
            misclf_type  = mtype,
            OvA          = cell(stat["OvA_d"], stat["OvA_p"]),
            OvR          = cell(stat["OvR_d"], stat["OvR_p"]),
            AvR          = cell(stat["AvR_d"], stat["AvR_p"]),
        ))

    # Column order & sorting
    df = pd.DataFrame(rows)
    df.sort_values(
        by=["dataset", "split", "wnum", "misclf_type"],
        key=lambda s: pd.Categorical(
            s, categories=(
                DATASETS if s.name == "dataset" else
                SPLITS   if s.name == "split"   else
                WN_LIST  if s.name == "wnum"    else
                MISCLF_TPS
            ),
            ordered=True), inplace=True)

    # Save per-metric table
    out_csv = f"exp-repair-4-1-8_wilcoxon_cliffs_{met_tag}.csv"
    df.to_csv(out_csv, index=False)
    print(f"[✓] {nice_name}  →  {out_csv}")
    
    per_metric_tables[met_tag] = df  # keep for merge

# ───────────────────────────────────────────────────────────────
# Integrated table (test split only), combining RR and BR results
# ───────────────────────────────────────────────────────────────
df_rr = per_metric_tables["RR"].query("split == 'test'").copy()
df_br = per_metric_tables["BR"].query("split == 'test'").copy()

# Disambiguate columns before merging
df_rr.rename(columns={"OvA":"RR_OvA","OvR":"RR_OvR","AvR":"RR_AvR"}, inplace=True)
df_br.rename(columns={"OvA":"BR_OvA","OvR":"BR_OvR","AvR":"BR_AvR"}, inplace=True)

# Merge on (dataset, wnum, misclf_type)
merge_cols = ["dataset","wnum","misclf_type"]
df_merge = (df_rr[merge_cols + ["RR_OvA","RR_OvR","RR_AvR"]]
            .merge(df_br[merge_cols + ["BR_OvA","BR_OvR","BR_AvR"]],
                   on=merge_cols, how="inner")
            .sort_values(["dataset","wnum","misclf_type"],
                         key=lambda s: pd.Categorical(
                             s, categories=(DATASETS if s.name=="dataset"
                                            else WN_LIST if s.name=="wnum"
                                            else MISCLF_TPS),
                             ordered=True)))

# Make "#weights" label explicit
df_merge.rename(columns={"wnum":"#weights"}, inplace=True)
df_merge["dataset"] = df_merge["dataset"].map({"c100": "C100", "tiny-imagenet": "TinyImg"})

# --- Pretty labels for pairwise comparisons ---
pair_label_map = {
    "RR_OvA": "Rep vs. AraW",
    "RR_OvR": "Rep vs. Rand",
    "RR_AvR": "AraW vs. Rand",
    "BR_OvA": "Rep vs. AraW",
    "BR_OvR": "Rep vs. Rand",
    "BR_AvR": "AraW vs. Rand",
}

# --- Reorder columns (then rename to pretty labels) ---
col_order = ["dataset", "#weights", "misclf_type",
             "RR_OvA", "RR_OvR", "RR_AvR", "BR_OvA", "BR_OvR", "BR_AvR"]

df_ordered = df_merge[col_order].rename(columns=pair_label_map)

out_all = "exp-repair-4-1-8_merged_test.csv"
# quoting=1 → quote all cells with double quotes in the CSV
df_ordered.to_csv(out_all, index=False, quoting=1)
print(f"[✓] wrote merged table → {out_all}")
