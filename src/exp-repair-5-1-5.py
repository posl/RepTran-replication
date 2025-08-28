import os, re, json, sys, math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Constants class provided by the library
from utils.constant import ViTExperiment

NUM_REPS = 5
# Requested alphas (in the original Arachne parameterization)
alpha_in_arachne_list = [1, 4, 8, 10]
# Convert to ratio form α/(1+α) used in our JSON filenames
alpha_ratio_list      = [a / (1 + a) for a in alpha_in_arachne_list]

def collect_records(ds: str, tgt_split: str) -> pd.DataFrame:
    """
    Scan JSON results and return a records DataFrame for a given dataset
    (c100 / tiny-imagenet) and split (repair / test).
    """
    big = []
    k_list        = [0]
    tgt_rank_list = [1, 2, 3]
    misclf_list   = ["src_tgt", "tgt_fp", "tgt_fn"]
    fl_method_li  = ["ours", "bl", "random"]
    w_num_list    = [236]  # see exp-repair-5.md

    for k in k_list:
        pretrained_dir = getattr(ViTExperiment, ds.replace("-", "_")).OUTPUT_DIR.format(k=k)
        for tgt_rank in tgt_rank_list:
            for misclf_full in misclf_list:
                save_dir = os.path.join(
                    pretrained_dir,
                    f"misclf_top{tgt_rank}",
                    f"{misclf_full}_repair_weight_by_de"   # this base path is sufficient
                )
                for flm in fl_method_li:
                    for alpha_raw, alpha_ratio in zip(alpha_in_arachne_list, alpha_ratio_list):
                        # exp id: 5-* when alpha_raw ∈ {1,4,8}, 4-* when alpha_raw == 10 (to match existing filenames)
                        exp_id = 4 if alpha_raw == 10 else 5
                        for w_num in w_num_list:
                            # --- setting_id (must match the JSON filename) -----------------------
                            parts = []
                            # Keep "n{wnum}" and "alpha{ratio}" to match the original file naming rule
                            parts.append(f"n{int(w_num) if w_num >= 1 else w_num}")
                            parts.append(f"alpha{alpha_ratio}")
                            parts.append("boundsArachne")
                            setting_id = "_".join(parts)

                            # --- reps loop ------------------------------------------------------
                            for reps in range(NUM_REPS):
                                jp = os.path.join(
                                    save_dir,
                                    f"exp-repair-{exp_id}-1-metrics_for_{tgt_split}_{setting_id}_{flm}_reps{reps}.json",
                                )
                                if not os.path.exists(jp):
                                    raise FileNotFoundError(jp)
                                with open(jp) as f:
                                    js = json.load(f)

                                # For test split, also fetch total repair time from the matching repair JSON
                                if tgt_split == "test":
                                    jp_repair = os.path.join(
                                        save_dir,
                                        f"exp-repair-{exp_id}-1-metrics_for_repair_{setting_id}_{flm}_reps{reps}.json",
                                    )
                                    with open(jp_repair) as f:
                                        js_repair = json.load(f)

                                big.append(
                                    {
                                        "dataset":   ds,
                                        "tgt. misclf. type": misclf_full,  # src_tgt / tgt_fp / tgt_fn
                                        "tgt_rank": tgt_rank,               # 1 / 2 / 3
                                        "method":    flm,
                                        "#modified weights": w_num,         # 236
                                        "alpha_raw": alpha_raw,             # 1 / 4 / 8 / 10
                                        "alpha":     alpha_ratio,           # 0.5 / 0.8 / … / 0.909…
                                        "reps_id": reps,
                                        "RR_tgt":    js.get("repair_rate_tgt"),
                                        "BR_all":    js.get("break_rate_overall"),
                                        "ΔACC.":     js.get("delta_acc"),
                                        "TOT_TIME": js_repair.get("tot_time") if tgt_split == "test" else js.get("tot_time"),
                                    }
                                )
    return pd.DataFrame(big)


def plot_alpha(df: pd.DataFrame, ds: str, tgt_split: str):
    """
    Draw line plots over α for RR, BR (and optionally ΔACC/TOT_TIME), with a single shared legend.
    """
    alpha_in_arachne_list = [1, 4, 8, 10]
    sns.set(style="whitegrid", font_scale=0.95)

    # Normalize method names for plotting
    replace = {"ours": "Ours", "bl": "Arachne", "random": "Random"}
    df["method"] = df["method"].map(replace)

    # Categorical x-axis with fixed α order
    alpha_cats = [str(a) for a in alpha_in_arachne_list]
    df["α_cat"] = pd.Categorical(df["alpha_raw"].astype(str), categories=alpha_cats, ordered=True)

    hue_order = ["Ours", "Arachne", "Random"]
    palette   = {"Ours": "C0", "Arachne": "C2", "Random": "C1"}
    metrics   = ["RR_tgt", "BR_all"]
    # Alternative metric sets:
    # metrics = ["RR_tgt", "BR_all", "TOT_TIME"]
    # metrics = ["RR_tgt", "BR_all", "ΔACC."]
    row_types = ["src_tgt", "tgt_fp", "tgt_fn"]
    marker_for_meth = {"Arachne": "v",  "Ours": "^",  "Random": "o"}
    
    base_size = 3
    fig, axes = plt.subplots(len(metrics), len(row_types), figsize=(base_size*4, base_size*2))

    for i, metric in enumerate(metrics):
        for j, mt in enumerate(row_types):
            subset = df[df["tgt. misclf. type"] == mt]
            ax = axes[i][j]
            if subset.empty:
                ax.axis("off")
                continue
            for meth in hue_order:
                sub_meth_df = subset[subset["method"] == meth]
                sns.lineplot(
                    data=sub_meth_df,
                    x="α_cat",
                    y=metric,
                    label=meth,
                    color=palette[meth],
                    marker=marker_for_meth[meth],
                    dashes=False,
                    sort=False,
                    ax=ax,
                    mec="black",
                    legend=False  # defer to the global legend
                )
            ax.set_title(f"{mt.upper().replace('_', '-')}", fontsize=12, fontstyle="italic")
            # --- x-axis ---
            ax.set_xlabel("")
            if i == len(metrics) - 1:
                ax.set_xlabel("$\\alpha$")

            # --- y-axis ---
            if j == 0:  # only left-most column shows the y-label
                if metric == "RR_tgt":
                    ax.set_ylabel("Repair Rate")
                elif metric == "BR_all":
                    ax.set_ylabel("Break Rate")
                elif metric == "ΔACC.":
                    ax.set_ylabel("ΔACC.")
            else:
                ax.set_ylabel("")
            ax.grid(True, linestyle="--", linewidth=0.5)

    # Shared legend (method-wise)
    legend_labels = {"Arachne": "ArachneW", "Ours": "REPTRAN", "Random": "Random"}
    handles = [
        Line2D([], [], marker=marker_for_meth[m], linestyle="", color=palette[m], 
               label=legend_labels[m], markersize=8, markerfacecolor=palette[m], markeredgecolor="black")
        for m in hue_order
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.1),
        ncol=3,
        frameon=True,
        fontsize=12
    )

    # Push the legend further down and make room for it
    fig.subplots_adjust(bottom=0.01, hspace=0.2, wspace=0.25)
    plt.tight_layout(rect=[0, 0.14, 1, 1])
    out_png = f"exp-repair-5-1_{ds}_{tgt_split}_alpha_lineplots.pdf"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    return out_png


def run_kruskal_over_alpha(df: pd.DataFrame, ds: str, tgt_split: str) -> pd.DataFrame:
    from scipy.stats import kruskal
    """
    Run Kruskal–Wallis tests over α for RR and BR, per dataset, pooling all methods.

    Parameters:
        df (pd.DataFrame): results from collect_records
        ds (str): "c100" or "tiny-imagenet"
        tgt_split (str): "repair" or "test"

    Returns:
        pd.DataFrame: H-statistic, p-value, and η² per dataset × split × metric
    """
    results = []
    metrics = {"RR_tgt": "RR", "BR_all": "BR"}

    for metric_col, metric_label in metrics.items():
        df_sub = df[df["dataset"] == ds]
        grouped = df_sub.groupby("alpha_raw")[metric_col].apply(list)

        if len(grouped) >= 2:  # need ≥2 groups
            try:
                h_val, p_val = kruskal(*grouped)
                n = df_sub.shape[0]
                k = len(grouped)
                eta2 = (h_val - k + 1) / (n - k) if n > k else float("nan")
            except Exception:
                h_val, p_val, eta2 = float("nan"), float("nan"), float("eta2")
        else:
            h_val, p_val, eta2 = float("nan"), float("nan"), float("nan")

        results.append({
            "Dataset": ds,
            "Split": tgt_split,
            "Metric": metric_label,
            "H-statistic": h_val,
            "p-value": p_val,
            "η²": eta2,
        })

    return pd.DataFrame(results)


# ╭───────────────── Entry point ─────────────────╮
if __name__ == "__main__":
    ds_list = ["c100", "tiny-imagenet"]
    tgt_split_list = ["test"]
    # If needed: tgt_split_list = ["repair", "test"]
    
    for ds_arg in ds_list:
        for tgt_split_arg in tgt_split_list:
            print(f"{'='*90}\nProcessing: dataset={ds_arg}, split={tgt_split_arg}")
            # --- Collect & save raw table -----------------------------------------
            df_all = collect_records(ds_arg, tgt_split_arg)
            out_csv = f"exp-repair-5-1_{ds_arg}_{tgt_split_arg}_results_all.csv"
            df_all.to_csv(out_csv, index=False)
            print("[INFO] saved", out_csv)

            # --- Plots ------------------------------------------------------------
            plot_alpha(df_all, ds_arg, tgt_split_arg)
            
            # --- Kruskal–Wallis over α -------------------------------------------
            kruskal_results = run_kruskal_over_alpha(df_all, ds_arg, tgt_split_arg)
            kruskal_out_csv = f"exp-repair-5-1_{ds_arg}_{tgt_split_arg}_kruskal_alpha.csv"
            kruskal_results.to_csv(kruskal_out_csv, index=False)
            print("[INFO] kruskal results saved to", kruskal_out_csv)
            print(kruskal_results)
            
            # ========= α vs RR / BR correlation (dataset × misclf_type × method) ==
            from scipy.stats import pearsonr

            def corr_by_method(df: pd.DataFrame, metric: str) -> pd.DataFrame:
                """metric = 'RR_tgt' or 'BR_all'"""
                res = (
                    df
                    .groupby(["dataset", "tgt. misclf. type", "method"], dropna=True)
                    .apply(lambda g: pearsonr(g["alpha_raw"], g[metric])[0] if len(g) > 1 else float("nan"))
                    .reset_index(name=f"corr_{metric}")
                )
                return res

            if tgt_split_arg == "test":
                # ① correlation between α and RR
                corr_rr = corr_by_method(df_all, "RR_tgt")
                # ② correlation between α and BR
                corr_br = corr_by_method(df_all, "BR_all")

                # Pivot to wide format
                tab_rr = (
                    corr_rr.pivot(index=["dataset", "tgt. misclf. type"],
                                  columns="method", values="corr_RR_tgt")
                          .rename(columns={"vmg":"Ours", "bl":"Arachne", "random":"Random"})
                )

                tab_br = (
                    corr_br.pivot(index=["dataset", "tgt. misclf. type"],
                                  columns="method", values="corr_BR_all")
                          .rename(columns={"vmg":"Ours", "bl":"Arachne", "random":"Random"})
                )

                # Concatenate RR and BR side-by-side and order columns
                table = pd.concat([tab_rr, tab_br], axis=1, keys=["RR", "BR"])
                table = table.reindex(columns=pd.MultiIndex.from_product(
                            [["RR","BR"], ["Ours","Arachne","Random"]]))

                # Save & print
                out_corr = f"exp-repair-5-1_{ds_arg}_{tgt_split_arg}_corr_alpha_full.csv"
                table.to_csv(out_corr)
                print("\n=== α vs RR / BR correlation table ===")
                print(table)
                print("[INFO] saved", out_corr)
