import os, json, math
from itertools import product

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Parameters you may edit
# ------------------------------------------------------------
DATASETS      = ["c100", "tiny-imagenet"]          # e.g. ["c100","tiny-imagenet"]
K             = 0                          # fold id
WN_LIST       = [236, 472, 944]
TGT_RANKS     = [1, 2, 3]
MISCLF_TYPES  = ["src_tgt", "tgt_fp", "tgt_fn"]
METHODS       = {"ours": "Ours", "bl": "Arachne", "random": "Random"}
REPS          = range(5)
ALPHA_STR     = 10/11
# adjust root template if your constant file differs
def root_dir(ds):
    # fallback when utils.constant is unavailable in this sandbox
    return f"./out_vit_{ds}_fold{K}"

# ------------------------------------------------------------
def load_json_safe(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        print("[MISS]", path)
        return None

def mean_ci(series):
    n = len(series)
    if n == 0:
        return np.nan, np.nan
    m = np.mean(series)
    ci = 1.96 * np.std(series, ddof=1) / math.sqrt(n)
    return m, ci

def collect_split(ds, split):
    rows = []
    base = root_dir(ds)
    for tgt_rank, mtype, wnum, rep_key, rep_id in product(
            TGT_RANKS, MISCLF_TYPES, WN_LIST, METHODS.keys(), REPS):
        save_dir = os.path.join(base,
                                f"misclf_top{tgt_rank}",
                                f"{mtype}_repair_weight_by_de")
        fname = (f"exp-repair-4-1-metrics_for_{split}_"
                 f"n{wnum}_alpha{ALPHA_STR}_boundsArachne_{rep_key}_reps{rep_id}.json")
        jpath = os.path.join(save_dir, fname)
        data  = load_json_safe(jpath)
        if data is None:
            continue
        rows.append({
            "ds": ds,
            "split": split,
            "wnum": wnum,
            "tgt_rank": tgt_rank,
            "misclf_type": mtype,
            "method": METHODS[rep_key],
            "rep_id": rep_id,
            "RR": data["repair_rate_tgt"],
            "BR": data["break_rate_overall"],
        })
    return pd.DataFrame(rows)

def stats_for_plot(df_long):
    out = []
    for (mtype, rank, wnum, method), g in df_long.groupby(
            ["misclf_type", "tgt_rank", "wnum", "method"]):
        rr = g["RR"];  br = g["BR"]
        out.append({
            "misclf_type": mtype, "tgt_rank": rank,
            "wnum": wnum, "method": method,
            "RR_mean": rr.mean(), "RR_min": rr.min(), "RR_max": rr.max(),
            "BR_mean": br.mean(), "BR_min": br.min(), "BR_max": br.max(),
        })
    return pd.DataFrame(out)

# ---------------------------------------
# 1. FacetGrid を描く汎用関数（RR か BR 専用）
# ---------------------------------------
def facet_plot_one(df_stats: pd.DataFrame, *, metric: str,
                   ds: str, split: str):
    """
    metric: "RR" または "BR" だけを片方の図に描く。
    • 色  : 手法ごと (Ours=C0, Arachne=C2, Random=C1)
    • 形  : 手法ごと (●, ■, ▲) — 塗りつぶし有り
    • シェード : 95% CI
    """
    import seaborn as sns, matplotlib.pyplot as plt

    sns.set(style="whitegrid", font_scale=0.95)

    palette = {"Ours": "C0", "Arachne": "C2", "Random": "C1"}
    marker_shape = {"Ours": "^", "Arachne": "v", "Random": "s"}
    marker_size  = {"Ours": 10,   "Arachne": 10, "Random": 7}
    methods = ["Ours", "Arachne", "Random"]
    x_offset = {"Ours": 18, "Arachne": -18, "Random": 0}
    label_map = {236: "236\n(0.005%)",
             472: "472\n(0.01%)",
             944: "944\n(0.02%)"}
    w_ticks   = sorted(df_stats["wnum"].unique())
    metric_to_name = {"RR": "Repair Rate", "BR": "Broken Rate"}

    g = sns.FacetGrid(df_stats,
                      row="misclf_type", col="tgt_rank",
                      height=2.5, aspect=1.2,
                      sharex=True, sharey=True)
    
    for meth in methods:
        sub = df_stats[df_stats["method"] == meth]

        for _, r in sub.iterrows():
            ax = g.axes[
                g.row_names.index(r["misclf_type"])
            ][
                g.col_names.index(r["tgt_rank"])
            ]
            # ★ シフトした x 座標を使う
            x_pos = r["wnum"] + x_offset[meth]

            mean = r[f"{metric}_mean"]
            ymin = r[f"{metric}_min"]
            ymax = r[f"{metric}_max"]

            ax.plot(x_pos, mean,
                    color=palette[meth],
                    marker=marker_shape[meth],
                    markersize=marker_size[meth],
                    linestyle='-', lw=1.4,
                    label=meth)
            ax.errorbar(x_pos, mean,
                        yerr=[[mean - ymin], [ymax - mean]],
                        fmt='none', ecolor=palette[meth],
                        elinewidth=2, capsize=9, capthick=1.2)

            ax.set_xticks(w_ticks)
            ax.set_xticklabels([label_map[w] for w in w_ticks], rotation=45)
            if metric == "RR":
                ax.set_ylim(-0.05, 1.05)
            else:
                assert metric == "BR", "metric must be 'RR' or 'BR'"
                ax.set_ylim(-0.005, 0.105)
            # ax.tick_params(labelsize=9)
    g.set_titles(row_template="Type: {row_name}", col_template="Top: {col_name}")
    g.set_axis_labels("#weights", metric_to_name[metric])
    for ax in g.axes.flat:
        axis_font_size = 10
        ax.xaxis.label.set_size(12)  # x軸ラベルのフォントサイズ
        ax.yaxis.label.set_size(12)  # y軸ラベルのフォントサイズ
        ax.tick_params(axis='both', labelsize=axis_font_size)  # x軸とy軸のtickのフォントサイズ
    g.fig.subplots_adjust(top=0.92, wspace=0.25, hspace=0.25, bottom=0.08)

    # 共通凡例（手法３本）
    handles, labels = [], []
    for ax in g.axes.flat:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh); labels.append(ll)
        if ax.get_legend(): ax.get_legend().remove()
    g.figure.legend(handles, labels, loc="upper center",
                    ncol=3, frameon=False, fontsize=9)

    out_png = f"exp-repair-4-1-7_{metric}_{ds}_{split}.png"
    g.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"[✓] saved {out_png}")

# -------------------- run all -------------------------
for ds in DATASETS:
    for split in ["repair","test"]:
        print(f"\nProcessing {ds} {split} ...")
        print("=" * 90)
        df_long = collect_split(ds, split)
        if df_long.empty:
            print(f"[WARN] no data for {ds}-{split}")
            continue
        df_stats = stats_for_plot(df_long)
        print(df_stats)
        # RR
        facet_plot_one(df_stats, metric="RR", ds=ds, split=split)
        # BR
        facet_plot_one(df_stats, metric="BR", ds=ds, split=split)
