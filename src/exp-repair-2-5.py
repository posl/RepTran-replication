#!/usr/bin/env python
# exp-repair-2-5.py  –  LoRA vs Arachne & Random summary

from __future__ import annotations       # 3.7–3.9 でも `|` 記法 OK
import os, re, json, sys
from itertools import product
from typing import Optional

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.constant import ViTExperiment, ExperimentRepair1, ExperimentRepair2

# ---------- 共有パラメータ ----------
DS          = "c100"
K_LIST      = [0]
TGT_RANKS   = [1, 2, 3]
MISCLF_TYPES = ["src_tgt", "tgt"]
FPFN_LIST   = [None, "fp", "fn"]
ALPHAS      = [0.2, 0.4, 0.6, 0.8]
NUM_REPS    = 5
LORA_RANK   = 16
HIDDEN_D    = 3072     # d_model of ViT layer 11
# -----------------------------------

# ════════════════════════════════════════════════════════════════════
# 1)  LoRA (exp-repair-2) ───────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════
_re_lora = re.compile(
    # r"exp-repair-2(?:-for_test)?-best_patch_r(\d+)_alpha_([\d\.]+)_reps(\d+)\.json"
    r"exp-repair-2(?:-for_test)?[_-]best_patch_r(\d+)_alpha_([\d\.]+)_reps(\d+)\.json"
)
def parse_lora(fname: str):
    m = _re_lora.match(fname)
    return (int(m[1]), float(m[2]), int(m[3])) if m else (None, None, None)

def row_from_lora(json_path: str,
                  tgt_rank: int,
                  misclf_type: str,
                  fpfn: Optional[str],
                  split: str):
    r, alpha, reps = parse_lora(os.path.basename(json_path))
    if r is None: return None
    with open(json_path) as f: data = json.load(f)
    return {
        "subject":           "ViT/C100",
        "method":            "LoRA",
        "tgt. misclf. type": misclf_type if fpfn is None else f"{misclf_type}_{fpfn}",
        "tgt_rank":          tgt_rank,
        "#modified weights": 2 * HIDDEN_D * r,
        "alpha":             alpha,
        "reps_id":           reps,
        "RR_tgt":            data.get("repair_rate_tgt"),
        "BR_all":            data.get("break_rate_overall"),
        "ΔACC.":             data.get("delta_acc"),
        "TOT_TIME":          data.get("tot_time") if split == "repair" else None,
    }

# ════════════════════════════════════════════════════════════════════
# 2)  Arachne / Random (exp-repair-1) ───────────────────────────────
#     − 1-5.py と同じパーサを簡略化して流用
# ════════════════════════════════════════════════════════════════════
_re_1 = re.compile(
    r"exp-repair-1-metrics_for_(?:repair|test)_(.+)_(bl|random)_reps(\d+)\.json"
)
def parse_1(fname: str):
    m = _re_1.match(fname)
    return (m[1], m[2], int(m[3])) if m else (None, None, None)

def setting_to_alpha(setting_id: str) -> Optional[float]:
    m = re.search(r"alpha([\d\.]+)", setting_id)
    return float(m[1]) if m else None

def row_from_1(json_path: str,
               tgt_rank: int,
               misclf_type: str,
               fpfn: Optional[str]):
    setting_id, fl_method, reps_id = parse_1(os.path.basename(json_path))
    if setting_id is None: return None
    with open(json_path) as f: data = json.load(f)

    method = "Arachne" if fl_method == "bl" else "Random"
    alpha  = setting_to_alpha(setting_id)
    n_val  = re.search(r"n([\d\.]+)", setting_id)
    n_val  = int(float(n_val[1])) if n_val else None
    num_w  = 8 * n_val * n_val if n_val else None   # Arachne/random ともに weights＝8n²

    return {
        "subject":           "ViT/C100",
        "method":            method,
        "tgt. misclf. type": misclf_type if fpfn is None else f"{misclf_type}_{fpfn}",
        "tgt_rank":          tgt_rank,
        "#modified weights": num_w,
        "alpha":             alpha,
        "reps_id":           reps_id,
        "RR_tgt":            data.get("repair_rate_tgt"),
        "BR_all":            data.get("break_rate_overall"),
        "ΔACC.":             data.get("delta_acc"),
        "TOT_TIME":          data.get("tot_time"),
    }

# ════════════════════════════════════════════════════════════════════
def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "repair"
    assert split in {"repair", "test"}

    records = []

    # ---------- 2-系 (LoRA) ----------
    for k, tgt_rank, misclf_type, fpfn, alpha in product(
            K_LIST, TGT_RANKS, MISCLF_TYPES, FPFN_LIST, ALPHAS):
        if misclf_type == "src_tgt" and fpfn is not None: continue
        root = getattr(ViTExperiment, DS).OUTPUT_DIR.format(k=k)
        if fpfn and misclf_type == "tgt":
            save_dir = os.path.join(root, f"misclf_top{tgt_rank}",
                                    f"{misclf_type}_{fpfn}_repair_weight_by_de")
        elif misclf_type == "all":
            save_dir = os.path.join(root, f"{misclf_type}_repair_weight_by_de")
        else:
            save_dir = os.path.join(root, f"misclf_top{tgt_rank}",
                                    f"{misclf_type}_repair_weight_by_de")

        for reps in range(NUM_REPS):
            name = (f"exp-repair-2-best_patch_r{LORA_RANK}_alpha_{alpha}_reps{reps}.json"
                    if split == "repair" else
                    f"exp-repair-2-for_test_best_patch_r{LORA_RANK}_alpha_{alpha}_reps{reps}.json")
            p = os.path.join(save_dir, name)
            if os.path.exists(p):
                rec = row_from_lora(p, tgt_rank, misclf_type, fpfn, split)
                if rec: records.append(rec)

    # ---------- 1-系 (Arachne / Random) ----------
    FL_METHODS = ["bl", "random"]
    EXP_LIST   = [ExperimentRepair1, ExperimentRepair2]  # 1-5 と同じ

    for k, tgt_rank, misclf_type, fpfn, fm, alpha, exp in product(
            K_LIST, TGT_RANKS, MISCLF_TYPES, FPFN_LIST,
            FL_METHODS, ALPHAS, EXP_LIST):

        if misclf_type == "src_tgt" and fpfn is not None: continue
        root = getattr(ViTExperiment, DS).OUTPUT_DIR.format(k=k)

        if fpfn and misclf_type == "tgt":
            save_dir = os.path.join(root, f"misclf_top{tgt_rank}",
                                    f"{misclf_type}_{fpfn}_repair_weight_by_de")
        elif misclf_type == "all":
            save_dir = os.path.join(root, f"{misclf_type}_repair_weight_by_de")
        else:
            save_dir = os.path.join(root, f"misclf_top{tgt_rank}",
                                    f"{misclf_type}_repair_weight_by_de")

        # setting_id は 1-5 仕様
        n = exp.NUM_IDENTIFIED_WEIGHTS
        parts = [f"n{n}", f"alpha{alpha}", "boundsArachne"]
        setting_id = "_".join(parts)

        for reps in range(NUM_REPS):
            name = f"exp-repair-1-metrics_for_{split}_{setting_id}_{fm}_reps{reps}.json"
            p = os.path.join(save_dir, name)
            if os.path.exists(p):
                rec = row_from_1(p, tgt_rank, misclf_type, fpfn)
                if rec: records.append(rec)

    if not records:
        sys.exit("No metrics found.")

    df = pd.DataFrame(records)
    out_prefix = f"exp-repair-2-5_{split}"

    df.to_csv(f"{out_prefix}_results_all.csv", index=False)
    print(f"[✓] big CSV → {out_prefix}_results_all.csv")

    # 数値列 cast
    for c in ["reps_id", "RR_tgt", "BR_all", "ΔACC.", "TOT_TIME"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # averaged
    gcols = ["method", "tgt. misclf. type", "#modified weights"]
    agg   = {"RR_tgt":"mean", "BR_all":"mean", "ΔACC.":"mean"}
    if split == "repair": agg["TOT_TIME"] = "mean"
    df.groupby(gcols, as_index=False).agg(agg) \
        .to_csv(f"{out_prefix}_results_averaged.csv", index=False)
    print(f"[✓] avg CSV  → {out_prefix}_results_averaged.csv")

    # ─── 可視化 ───
    sns.set(style="whitegrid")
    palette = {"LoRA":"C0", "Arachne":"C2", "Random":"C1"}
    order   = ["LoRA", "Arachne", "Random"]
    metrics = ["RR_tgt", "BR_all", "ΔACC."] + (["TOT_TIME"] if split=="repair" else [])
    sub_types = ["src_tgt", "tgt_fp", "tgt_fn", "tgt"]

    # (C) violin
    fig_v, ax_v = plt.subplots(len(sub_types), len(metrics), figsize=(18, 12))
    for i, mt in enumerate(sub_types):
        for j, met in enumerate(metrics):
            sub = df[df["tgt. misclf. type"] == mt]
            ax = ax_v[i, j]
            if sub.empty: ax.set_visible(False); continue
            sns.violinplot(data=sub, x="method", y=met,
                           order=order, palette=palette,
                           cut=0, scale="width", inner="box", ax=ax)
            ax.set_title(f"{mt} – {met}", fontsize=11)
            ax.set_xlabel(""); ax.set_ylabel(met, fontsize=9)
    plt.tight_layout(); plt.savefig(f"{out_prefix}_violin.png", dpi=150)

    # (D) alpha-transition
    fig_l, ax_l = plt.subplots(len(sub_types), len(metrics), figsize=(18, 14))
    for i, mt in enumerate(sub_types):
        for j, met in enumerate(metrics):
            ax = ax_l[i, j]
            sub = df[(df["tgt. misclf. type"] == mt) & df["alpha"].notnull()]
            if sub.empty: ax.set_visible(False); continue
            sub.sort_values("alpha", inplace=True)
            sns.lineplot(data=sub, x="alpha", y=met,
                         hue="method", hue_order=order,
                         palette=palette, marker="o", dashes=False, ax=ax)
            ax.set_title(f"{mt} – {met}", fontsize=11)
            ax.set_xlabel("alpha", fontsize=9); ax.set_ylabel(met, fontsize=9)
            ax.set_xticks(ALPHAS); ax.set_xlim(min(ALPHAS), max(ALPHAS))
    plt.tight_layout(); plt.savefig(f"{out_prefix}_alpha_lineplots.png", dpi=150)
    print(f"[✓] plots saved (violin / line) under {out_prefix}_*.png")

if __name__ == "__main__":
    main()
