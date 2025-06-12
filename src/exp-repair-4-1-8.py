#!/usr/bin/env python
# rq2_stats_table_rr_br.py  –  RQ-2: weight-variant experiments (RR / BR)

import json, math, itertools, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

# ──────────── 実験設定 ──────────────────────────────────────────
DATASETS     = ["c100", "tiny-imagenet"]   # データセット
SPLITS       = ["repair", "test"]          # 評価 split
K            = 0                           # fold id
TGT_RANKS    = [1, 2, 3]                   # 誤分類ランキング
MISCLF_TPS   = ["src_tgt", "tgt_fp", "tgt_fn"]
WN_LIST      = [236, 472, 944]             # 3 通りの重み数
REPS         = range(5)                    # 5 回リピート
ALPHA_STR    = 10/11
ROOT_TMPL    = "/src/src/out_vit_{ds}_fold{K}"

# 3 手法と JSON ファイル名中のキー
METHODS = {"ours": "ours", "arachne": "bl", "random": "random"}
PAIRS   = [("ours", "arachne"),
           ("ours", "random"),
           ("arachne", "random")]

METRIC_INFO = dict(
    RR=("repair_rate_tgt",  "Repair Rate"),
    BR=("break_rate_overall",   "Break  Rate"),
)

# ──────────── JSON ⇒ 値 ────────────────────────────────────────
def metric_value(ds, split, mtype, rank, rep,
                 wnum, meth_key, json_key):
    """1 個の JSON を開き，目的メトリクスを返す"""
    base = Path(ROOT_TMPL.format(ds=ds, K=K))
    jdir = base / f"misclf_top{rank}" / f"{mtype}_repair_weight_by_de"
    fn = (f"exp-repair-4-1-metrics_for_{split}_"
          f"n{wnum}_alpha{ALPHA_STR}_boundsArachne_{meth_key}_reps{rep}.json")
    with open(jdir / fn) as f:
        return json.load(f)[json_key]

# ──────────── Wilcoxon & Cliff’s Δ (対応あり) ────────────────
def paired_cliffs_delta(v1: np.ndarray, v2: np.ndarray) -> float:
    diff = v1 - v2
    return (np.sum(diff > 0) - np.sum(diff < 0)) / diff.size

def wilcoxon_block(values: dict) -> dict:
    """
    values = {method: np.array(15)}
    → Holm 補正後 p 値 & Cliff’s Δ（符号付き）
    """
    out, p_raw = {}, []
    for m1, m2 in PAIRS:
        v1, v2 = values[m1], values[m2]
        # p 値 (同値なら 1.0)
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
    # Holm
    _, p_adj, _, _ = multipletests(p_raw, method="holm")
    for (m1, m2), p_c in zip(PAIRS, p_adj):
        out[f"{m1[:1].upper()}v{m2[:1].upper()}_p"] = p_c
    return out

def stars(p):
    return "***" if p <= .001 else "**" if p <= .01 else "*" if p <= .05 else ""

def cell(d, p):
    return f"'{d:+.2f} {stars(p)}"

# ──────────── main ────────────────────────────────────────────
per_metric_tables = {}            # ★ 追加：あとでマージするため保持
for met_tag, (json_key, nice_name) in METRIC_INFO.items():
    rows = []

    for ds, split, wnum, mtype in itertools.product(
            DATASETS, SPLITS, WN_LIST, MISCLF_TPS):

        # 15 点 × 3 手法を収集
        vals = {m: [] for m in METHODS}
        for rank, rep in itertools.product(TGT_RANKS, REPS):
            for m, key in METHODS.items():
                vals[m].append(metric_value(ds, split, mtype,
                                            rank, rep, wnum,
                                            key, json_key))
        vals = {m: np.asarray(v) for m, v in vals.items()}

        stat = wilcoxon_block(vals)
        rows.append(dict(
            dataset      = ds,
            split        = split,
            wnum         = wnum,
            misclf_type  = mtype,
            OvA          = cell(stat["OvA_d"], stat["OvA_p"]),
            OvR          = cell(stat["OvR_d"], stat["OvR_p"]),
            AvR          = cell(stat["AvR_d"], stat["AvR_p"]),
        ))

    # 列順 & 並び替え
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
            ordered=True)).reset_index(drop=True, inplace=True)

    # 保存
    out_csv = f"exp-repair-4-1-8_wilcoxon_cliffs_{met_tag}.csv"
    df.to_csv(out_csv, index=False)
    print(f"[✓] {nice_name}  →  {out_csv}")
    
    per_metric_tables[met_tag] = df          # ★ 保存
    
# ─────────────────────────────────────────────
# ★ ここから統合テーブル（split == test だけ）
# ─────────────────────────────────────────────
df_rr = per_metric_tables["RR"].query("split == 'test'").copy()
df_br = per_metric_tables["BR"].query("split == 'test'").copy()

# 列名を衝突しないように付け替え
df_rr.rename(columns={"OvA":"RR_OvA","OvR":"RR_OvR","AvR":"RR_AvR"}, inplace=True)
df_br.rename(columns={"OvA":"BR_OvA","OvR":"BR_OvR","AvR":"BR_AvR"}, inplace=True)

# マージ（dataset, wnum, misclf_type で結合）
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

# 見出し #weights をラベル付きにしても OK
df_merge.rename(columns={"wnum":"#weights"}, inplace=True)

out_all = "exp-repair-4-1-8_merged_test.csv"
df_merge.to_csv(out_all, index=False, quoting=1)   # quoting=1 → 全セルを "…" で囲む
print(f"[✓] wrote merged table → {out_all}")