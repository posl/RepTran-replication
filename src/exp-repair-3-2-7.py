#!/usr/bin/env python
# rq1_stats_table_rr_br.py  –  RR / BR  の Wilcoxon＋Holm＋Cliff’s Δ

import os, json, math, itertools, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

# ────── 固定パラメータ ──────────────────────────────────────────
DATASETS   = ["c100", "tiny-imagenet"]
SPLITS     = ["repair", "test"]
K          = 0
TGT_RANKS  = [1, 2, 3]
MISCLF_TPS = ["src_tgt", "tgt_fp", "tgt_fn"]
REPS       = range(5)

ALPHA      = 10/11
ALPHA_STR  = f"alpha{ALPHA}_boundsArachne"
ROOT_TMPL  = "/src/src/out_vit_{ds}_fold{K}"

METHODS = {"ours": "ours", "arachne": "bl", "random": "random"}
PAIRS   = [("ours", "arachne"), ("ours", "random"), ("arachne", "random")]

METRIC_INFO = dict(
    RR=("repair_rate_tgt",  "Repair Rate"),
    BR=("break_rate_overall",   "Break  Rate"),
)

# ────── JSON 読み出し ───────────────────────────────────────────
def metric_value(ds, split, mtype, rank, rep, method_key, json_key):
    base = Path(ROOT_TMPL.format(ds=ds, K=K))
    jdir = base / f"misclf_top{rank}" / f"{mtype}_repair_weight_by_de"
    if method_key in {"ours", "random"}:
        fn = f"exp-repair-3-2-metrics_for_{split}_{ALPHA_STR}_{method_key}_reps{rep}.json"
    else:           # arachne (bl)
        fn = f"exp-repair-3-1-metrics_for_{split}_{ALPHA_STR}_bl_reps{rep}.json"
    with open(jdir / fn) as f:
        return json.load(f)[json_key]

# ────── Wilcoxon & Cliff’s Δ (対応あり) ──────────────────────────
def paired_cliffs_delta(v1: np.ndarray, v2: np.ndarray):
    """対応あり Cliff’s Δ  =  (n_pos - n_neg) / N"""
    diff = v1 - v2
    n_pos = np.sum(diff > 0)
    n_neg = np.sum(diff < 0)
    return (n_pos - n_neg) / diff.size if diff.size else 0.0

def wilcoxon_block(values):
    """values = {method: np.array(15)}   ->   {OvA_p, OvA_d, …}"""
    out = {}
    p_raw = []
    # 生 p と Δ をまず計算
    for m1, m2 in PAIRS:
        v1, v2 = values[m1], values[m2]
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

    # Holm 補正
    _, p_adj, _, _ = multipletests(p_raw, method="holm")
    for (m1, m2), p_c in zip(PAIRS, p_adj):
        tag = f"{m1[:1].upper()}v{m2[:1].upper()}"
        out[f"{tag}_p"] = p_c
    return out

def star(p):
    return "***" if p <= .001 else "**" if p <= .01 else "*" if p <= .05 else ""

def cell(d, p):
    return f"'{d:+.2f} {star(p)}"      # +0.45 ** のように符号を残す

# ────── main ────────────────────────────────────────────────────
for metric_tag, (json_key, nice_name) in METRIC_INFO.items():
    rows = []
    for ds, split, mtype in itertools.product(DATASETS, SPLITS, MISCLF_TPS):
        # 15 データ点 × 3 手法
        vals = {m: [] for m in METHODS}
        for rank, rep in itertools.product(TGT_RANKS, REPS):
            for m, key in METHODS.items():
                vals[m].append(
                    metric_value(ds, split, mtype, rank, rep, key, json_key)
                )
        vals = {m: np.array(v) for m, v in vals.items()}

        stat = wilcoxon_block(vals)
        rows.append({
            "dataset": ds,
            "split": split,
            "misclf_type": mtype,
            "OvA": cell(stat["OvA_d"], stat["OvA_p"]),
            "OvR": cell(stat["OvR_d"], stat["OvR_p"]),
            "AvR": cell(stat["AvR_d"], stat["AvR_p"]),
        })

    # 並べ替え & 保存
    order = dict(dataset=DATASETS, split=SPLITS,
                 misclf_type=MISCLF_TPS)
    df = pd.DataFrame(rows).sort_values(
        ["dataset", "split", "misclf_type"],
        key=lambda s: s.map({v: i for col in ["dataset","split","misclf_type"]
                                   for i,v in enumerate(order[col])})
    )
    out_csv = f"exp-repair-3-2-7_wilcoxon_cliffs_{metric_tag}.csv"
    df.to_csv(out_csv, index=False)
    print(f"[✓] {nice_name}  →  {out_csv}")
