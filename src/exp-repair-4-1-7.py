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
SPLIT_LIST    = ["test"]
K             = 0                          # fold id
WN_LIST       = [11, 236, 472, 944]
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
        if wnum != 11:
            fname = (f"exp-repair-4-1-metrics_for_{split}_"
                 f"n{wnum}_alpha{ALPHA_STR}_boundsArachne_{rep_key}_reps{rep_id}.json")
        else:
            # rep_key (手法名) ごとに微妙にファイル名がちゃうねんな
            if rep_key == "bl":
                fname = f"exp-repair-3-1-metrics_for_{split}_alpha{ALPHA_STR}_boundsArachne_{rep_key}_reps{rep_id}.json"
            elif rep_key == "ours" or rep_key == "random":
                fname = f"exp-repair-3-2-metrics_for_{split}_alpha{ALPHA_STR}_boundsArachne_{rep_key}_reps{rep_id}.json"
            else:
                raise NotImplementedError
                
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
        
        rr_mean, rr_ci = mean_ci(rr)
        br_mean, br_ci = mean_ci(br)

        out.append({
            "misclf_type": mtype, "tgt_rank": rank,
            "wnum": wnum, "method": method,
            "RR_mean": rr_mean, "RR_ci": rr_ci,
            "BR_mean": br_mean, "BR_ci": br_ci,
        })
        # out.append({
        #     "misclf_type": mtype, "tgt_rank": rank,
        #     "wnum": wnum, "method": method,
        #     "RR_mean": rr.mean(), "RR_min": rr.min(), "RR_max": rr.max(),
        #     "BR_mean": br.mean(), "BR_min": br.min(), "BR_max": br.max(),
        # })
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
    label_map = {
             11: "11",
             236: "236\n(0.005%)",
             472: "472\n(0.01%)",
             944: "944\n(0.02%)"}
    w_ticks   = sorted(df_stats["wnum"].unique())
    metric_to_name = {"RR": "Repair Rate", "BR": "Break Rate"}

    g = sns.FacetGrid(df_stats,
                      row="misclf_type", col="tgt_rank",
                      height=1.5, aspect=1.8,
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
            # ymin = r[f"{metric}_min"]
            # ymax = r[f"{metric}_max"]
            ci   = r[f"{metric}_ci"]


            ax.plot(x_pos, mean,
                    color=palette[meth],
                    marker=marker_shape[meth],
                    markersize=marker_size[meth],
                    linestyle='-', lw=1.4,
                    label=meth)
            # ax.errorbar(x_pos, mean,
            #             yerr=[[mean - ymin], [ymax - mean]],
            #             fmt='none', ecolor=palette[meth],
            #             elinewidth=2, capsize=9, capthick=1.2)
            ax.errorbar(x_pos, mean,
                        yerr=ci,
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
        axis_font_size = 8
        ax.xaxis.label.set_size(10)  # x軸ラベルのフォントサイズ
        ax.yaxis.label.set_size(10)  # y軸ラベルのフォントサイズ
        ax.tick_params(axis='both', labelsize=axis_font_size)  # x軸とy軸のtickのフォントサイズ
    g.fig.subplots_adjust(top=0.9, wspace=0.25, hspace=0.33, bottom=0.08)

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

    out_file_name = f"exp-repair-4-1-7_{metric}_{ds}_{split}"
    out_pdf = f"{out_file_name}.pdf"
    g.savefig(out_pdf, dpi=300, bbox_inches="tight")
    print(f"[✓] saved {out_pdf}")

def facet_bar_one(df_stats: pd.DataFrame, *, metric: str,
                  ds: str, split: str):
    """
    棒グラフ版 – 既存スタイルに合わせて
    """
    import seaborn as sns, matplotlib.pyplot as plt

    sns.set(style="whitegrid", font_scale=0.95)

    palette = {
        "Ours":    "#4E79A7",   # 深めのブルー
        "Arachne": "#E15759",   # 落ち着いたコーラル
        "Random":  "#76B7B2",   # グレイッシュなグリーン
    }
    label_map = {11: "11",
                 236: "236\n(0.005%)",
                 472: "472\n(0.01%)",
                 944: "944\n(0.02%)"}
    w_ticks   = sorted(df_stats["wnum"].unique())
    metric_to_name = {"RR": "Repair Rate", "BR": "Break Rate"}
    methods_order = ["Arachne", "Ours", "Random"]

    g = sns.FacetGrid(df_stats,
                      row="misclf_type", col="tgt_rank",
                      height=1.5, aspect=1.8,
                      sharex=True, sharey=True)

    # ── 棒を描く ──
    g.map_dataframe(
        sns.barplot,
        x="wnum", y=f"{metric}_mean",
        hue="method", palette=palette,
        hue_order=methods_order,
        linewidth=0,            # 枠線なし
        dodge=True
    )

    # ── CI をバーの上端だけに描く ──
    for ax, (_, sub) in zip(g.axes.flat, df_stats.groupby(["misclf_type", "tgt_rank"])):
        for i, w in enumerate(w_ticks):
            for j, meth in enumerate(methods_order):
                row = sub[(sub["wnum"] == w) & (sub["method"] == meth)]
                if row.empty:           # データなし
                    continue
                mean = row[f"{metric}_mean"].values[0]
                ci   = row[f"{metric}_ci"  ].values[0]
                # dodge 量は seaborn の内部計算に合わせる（幅 0.8 の ±)
                xpos = i - 0.8/2 + (j+0.5)*(0.8/3)
                ax.errorbar(x=xpos, y=mean,
                            yerr=ci,
                            fmt='none', capsize=0, capthick=1.1,
                            ecolor="black", elinewidth=1.2)

    # 軸体裁
    for ax in g.axes.flat:
        ax.set_xticks(range(len(w_ticks)))
        ax.set_xticklabels([label_map[w] for w in w_ticks], rotation=45)
        if metric == "RR":
            ax.set_ylim(0, 1.05)
        else:
            ax.set_ylim(0, 0.105)

    g.set_titles(row_template="Type: {row_name}", col_template="Top: {col_name}", size=10)
    g.set_axis_labels("#weights", metric_to_name[metric])
    for ax in g.axes.flat:
        axis_font_size = 8
        ax.xaxis.label.set_size(10)  # x軸ラベルのフォントサイズ
        ax.yaxis.label.set_size(10)  # y軸ラベルのフォントサイズ
        ax.tick_params(axis='both', labelsize=axis_font_size)  # x軸とy軸のtickのフォントサイズ
    g.fig.subplots_adjust(top=0.88, wspace=0.25, hspace=0.4, bottom=0.08)
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

    stem = f"exp-repair-4-1-7_{metric}_{ds}_{split}"
    g.savefig(f"{stem}_bar.pdf", dpi=300, bbox_inches="tight")
    print(f"[✓] saved {stem}_bar.pdf")
    
# ============================================================
#  RR × BR 散布図  ── 色=method / 形=method  （重み数ごとに別ファイル）
# ============================================================
def scatter_rr_br(df_long, ds, split, *, wnum_filter=None):
    """
    wnum_filter: 11 / 236 / 472 / 944 のいずれかを渡すと，
                 その重み数だけで散布図を描く。
    """
    import seaborn as sns, matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # ------ フィルタリング ------
    if wnum_filter is not None:
        df_long = df_long[df_long["wnum"] == wnum_filter]

    # ===== ピボットして平均を取る（rep_id ごとに集約） =====
    keep = ["method", "wnum", "tgt_rank", "misclf_type", "rep_id"]
    agg  = (df_long.groupby(keep)
                     .agg(RR=("RR", "mean"), BR=("BR", "mean"))
                     .reset_index())

    # ------ 描画設定 ------
    sns.set(style="whitegrid", font_scale=0.95)
    color_for_meth  = {"Arachne": "C2", "Ours": "C0", "Random": "C1"}
    marker_for_meth = {"Arachne": "v",  "Ours": "^",  "Random": "o"}
    size_for_meth   = {"Arachne": 50,   "Ours": 50,   "Random": 50}

    fig, ax = plt.subplots(figsize=(4.8, 4.8))

    for (meth, _), sub in agg.groupby(["method", "wnum"], sort=False):
        ax.scatter(sub["BR"], sub["RR"],
                   color=color_for_meth[meth],
                   marker=marker_for_meth[meth],
                   s=size_for_meth[meth],
                   edgecolors="black", linewidths=0.5, alpha=0.75,
                   label=meth)

    # 軸・凡例体裁
    ax.set_ylim(-0.05, 1.05)
    # ax.set_xlim(-0.005, 0.065)
    ax.set_xlim(-0.000, 0.060)
    ax.set_xlabel("Break Rate", fontsize=10)
    ax.set_ylabel("Repair Rate", fontsize=10)
    title_w = f" – {wnum_filter} weights" if wnum_filter else ""
    fig.suptitle(f"{ds} – {split}{title_w}", fontsize=10, y=0.98)

    # 共通凡例
    handles, labels = [], []
    for m in ["Arachne", "Ours", "Random"]:
        handles.append(Line2D([], [], marker=marker_for_meth[m],
                              color=color_for_meth[m],
                              markersize=7, linestyle="",
                              label=m))
        labels.append(m)
    ax.legend(handles, labels, loc="lower right", frameon=True, fontsize=10)

    # --------------- 保存 ---------------
    stem = f"exp-repair-4-1-7_scatter_{ds}_{split}_w{wnum_filter}"
    fig.savefig(f"{stem}.pdf", dpi=300, bbox_inches="tight")
    print(f"[✓] saved {stem}.pdf")
    
# ============================================================
#  RR × BR 散布図（236 / 472 / 944 を横に並べたパネル）
# ============================================================
def scatter_rr_br_panel(df_long, ds, split, *, wnums=(236, 472, 944)):
    """
    • 横方向に len(wnums) 枚並べ，1 つの PDF に保存
    • 色・形は method 固定
    """
    import seaborn as sns, matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import math

    def nice_xmax(value, step=0.01, margin=0.001):
        return step * math.ceil((value + margin) / step)

    # ---------- 共通スタイル ----------
    sns.set(style="whitegrid", font_scale=0.95)
    color_for_meth  = {"Arachne": "C2", "Ours": "C0", "Random": "C1"}
    marker_for_meth = {"Arachne": "v",  "Ours": "^",  "Random": "o"}
    size_for_meth   = {"Arachne": 50,   "Ours": 50,   "Random": 50}

    n_cols = len(wnums)
    base_size = 3.2
    fig, axes = plt.subplots(1, n_cols, figsize=(base_size*n_cols, base_size), sharex=True, sharey=True)
    
    # df_longのBRのマックスを取得
    max_br = df_long["BR"].max()
    x_max  = nice_xmax(max_br, step=0.01)   # ← ここで上限を決定

    # ---------------------------------------------------------
    for ax, wnum in zip(axes, wnums):
        sub_df = df_long[df_long["wnum"] == wnum]

        # rep_id 単位で平均 (散布図が見やすくなる)
        keep = ["method", "wnum", "tgt_rank", "misclf_type", "rep_id"]
        agg  = (sub_df.groupby(keep)
                        .agg(RR=("RR", "mean"), BR=("BR", "mean"))
                        .reset_index())

        for meth, g in agg.groupby("method"):
            ax.scatter(g["BR"], g["RR"],
                       color=color_for_meth[meth],
                       marker=marker_for_meth[meth],
                       s=size_for_meth[meth],
                       edgecolors="black", linewidths=0.5, alpha=0.75)

        ax.set_title(f"$N_w = {wnum}$", fontsize=11)
        # if wnum == 11:
        #     ax.set_title(f"$N_w = {wnum}$", fontsize=11)
        # else:
        #     if wnum == 236:
        #         ratio = 0.005
        #     elif wnum == 472:
        #         ratio = 0.01
        #     elif wnum == 944:
        #         ratio = 0.02
        #     ax.set_title(f"$N_w = {wnum}$ ({ratio}%)", fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, x_max)   # 必要なら調整
        ax.grid(True, linestyle="--", linewidth=0.5)

    # ---- 軸ラベルは外側だけ ----
    axes[0].set_ylabel("Repair Rate", fontsize=11)
    for ax in axes:
        ax.set_xlabel("Break Rate", fontsize=11)

    # fig.suptitle(f"{ds} – {split}", fontsize=11, y=0.95)

    # ---- 共通凡例 ----
    legend_labels = {"Arachne": "ArachneW", "Ours": "REPTRAN", "Random": "Random"}
    handles = [
        Line2D([], [], marker=marker_for_meth[m], linestyle="",
            color=color_for_meth[m], markersize=7, label=legend_labels[m], markeredgecolor="black")
        for m in ["Arachne", "Ours", "Random"]
    ]
    # fig.legend(handles=handles, loc="lower center", ncol=3,
    #            frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.04))

    # fig.subplots_adjust(wspace=0.15, bottom=0.18)
    # ── パネル間の横スペースを広げる
    fig.subplots_adjust(wspace=0.25,   # ← ここを 0.15 ⇒ 0.25 など好みで
                        bottom=0.22)   # ← 凡例をさらに下へ押し出す

    # ── 凡例を下げる
    fig.legend(handles=handles, loc="lower center",
            ncol=3, frameon=True, fontsize=10,
            bbox_to_anchor=(0.5, -0.07))  # ← -0.04 ⇒ -0.07 など

    # ---- 保存 ----
    stem = f"exp-repair-4-1-7_scatterPanel_{ds}_{split}"
    fig.savefig(f"{stem}.pdf", dpi=300, bbox_inches="tight")
    print(f"[✓] saved {stem}.pdf")


# ============================================================
#  RR–BR Pearson correlation (per dataset × split × method)
# ============================================================
def corr_rr_br_by_split(df_long):
    import numpy as np
    import pandas as pd

    # データをピボットして、methodごとにRRとBRを抽出できるようにする
    df_pivot = df_long.pivot(
        index=["ds", "split", "wnum", "tgt_rank", "misclf_type", "rep_id"],
        columns="method",
        values=["RR", "BR"]
    )

    methods = ["Arachne", "Ours", "Random"]
    rows = []

    # ds × split ごとに処理
    for (ds, split), sub in df_pivot.groupby(level=["ds", "split"]):
        row = {"dataset": ds, "split": split}
        for method in methods:
            try:
                rr = sub["RR"][method]
                br = sub["BR"][method]
                corr = np.corrcoef(rr, br)[0, 1]
                row[method] = f"{corr:.3f}"
            except Exception:
                row[method] = "n/a"
        rows.append(row)

    df_corr = pd.DataFrame(rows)
    # df_corrをcsvで保存
    out_csv = f"exp-repair-4-1-7_corr_rr_br_by_split.csv"
    df_corr.to_csv(out_csv, index=False)
    print(f"[✓] saved {out_csv}")
    return df_corr

def harmonic_mean(rr, br):
    if rr + (1 - br) == 0:
        return 0
    return 2 * rr * (1 - br) / (rr + (1 - br))

def output_harmonic_summary(df_long):
    df = df_long.copy()
    print(df)
    df["Dataset"] = df["ds"].map({"c100": "C100", "tiny-imagenet": "TinyImg"})
    df["Method"] = df["method"].map({"Ours": "RepTran", "Arachne": "Arachne", "Random": "Random"})
    df["Harmonic"] = df.apply(lambda row: harmonic_mean(row["RR"], row["BR"]), axis=1)

    harmonic_summary = (
        df.groupby(["Dataset", "wnum", "Method"])["Harmonic"]
        .mean()
        .reset_index()
        .pivot(index=["Dataset", "wnum"], columns="Method", values="Harmonic")
        .reset_index()
    )
    harmonic_summary = harmonic_summary[["Dataset", "wnum", "RepTran", "Arachne", "Random"]]
    harmonic_summary.columns = ["Dataset", "N_w", "RepTran", "Arachne", "Random"]
    harmonic_summary.to_csv("./exp-repair-4-1-7_harmonic_mean_summary.csv", index=False)
    print("[✓] saved harmonic_mean_summary.csv")
    print(harmonic_summary)

# -------------------- run all -------------------------
for ds in DATASETS:
    for split in SPLIT_LIST:
    # for split in ["repair","test"]:
        print(f"\nProcessing {ds} {split} ...")
        print("=" * 90)
        df_long = collect_split(ds, split)
        # ★ 横 3 連のパネルを生成 ★
        scatter_rr_br_panel(df_long, ds, split, wnums=WN_LIST)
        # 重み数ごとに 3 枚の散布図を出力
        # for w in WN_LIST:             # 236, 472, 944
        #     scatter_rr_br(df_long, ds, split, wnum_filter=w)
        if df_long.empty:
            print(f"[WARN] no data for {ds}-{split}")
            continue
        df_stats = stats_for_plot(df_long)
        print(df_stats)
        # RR
        facet_plot_one(df_stats, metric="RR", ds=ds, split=split)
        # BR
        facet_plot_one(df_stats, metric="BR", ds=ds, split=split)
        
        facet_bar_one(df_stats, metric="RR", ds=ds, split=split)
        facet_bar_one(df_stats, metric="BR", ds=ds, split=split)
# ② 相関テーブル（DS × split 単位）
all_long = pd.concat([collect_split(d, s)
                      for d in DATASETS
                      for s in ["repair", "test"]],
                     ignore_index=True)
corr_rr_br_by_split(all_long)
# 調和平均テーブル出力
output_harmonic_summary(all_long)