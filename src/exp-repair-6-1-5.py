"""
exp-repair-6-1-5.py  —  visualization for p sensitivity experiment (RQ5)

Collects JSON results and draws line plots over p values.
Also runs Kruskal-Wallis test to check statistical significance.
"""
import os, json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import kruskal

from utils.constant import ViTExperiment

NUM_REPS         = 5
FIXED_WNUM       = 236
ALPHA_IN_ARACHNE = 10
FIXED_ALPHA      = ALPHA_IN_ARACHNE / (1 + ALPHA_IN_ARACHNE)
FIXED_BOUNDS     = "Arachne"
P_LIST           = [0.1, 0.9]


def collect_records(ds: str, tgt_split: str) -> pd.DataFrame:
    """Scan JSON results and return a DataFrame for a given dataset and split."""
    records = []
    k_list        = [0]
    tgt_rank_list = [1, 2, 3]
    misclf_list   = ["src_tgt", "tgt_fp", "tgt_fn"]
    fl_method_li  = ["ours", "bl", "random"]

    for k in k_list:
        pretrained_dir = getattr(ViTExperiment, ds.replace("-", "_")).OUTPUT_DIR.format(k=k)
        for tgt_rank in tgt_rank_list:
            for misclf_full in misclf_list:
                save_dir = os.path.join(
                    pretrained_dir,
                    f"misclf_top{tgt_rank}",
                    f"{misclf_full}_repair_weight_by_de",
                )
                for p in P_LIST:
                    setting_id = f"n{FIXED_WNUM}_alpha{FIXED_ALPHA}_bounds{FIXED_BOUNDS}_p{p}"
                    for flm in fl_method_li:
                        for reps in range(NUM_REPS):
                            # Test metrics
                            jp = os.path.join(
                                save_dir,
                                f"exp-repair-6-1-metrics_for_{tgt_split}_{setting_id}_{flm}_reps{reps}.json",
                            )
                            if not os.path.exists(jp):
                                raise FileNotFoundError(jp)
                            with open(jp) as f:
                                js = json.load(f)

                            # Repair metrics (for tot_time)
                            jp_repair = os.path.join(
                                save_dir,
                                f"exp-repair-6-1-metrics_for_repair_{setting_id}_{flm}_reps{reps}.json",
                            )
                            with open(jp_repair) as f:
                                js_repair = json.load(f)

                            records.append({
                                "dataset":    ds,
                                "misclf_type": misclf_full,
                                "tgt_rank":   tgt_rank,
                                "method":     flm,
                                "p":          p,
                                "reps_id":    reps,
                                "RR_tgt":     js.get("repair_rate_tgt"),
                                "BR_all":     js.get("break_rate_overall"),
                                "delta_acc":  js.get("delta_acc"),
                                "tot_time":   js_repair.get("tot_time"),
                            })
    return pd.DataFrame(records)


def plot_p_sensitivity(df: pd.DataFrame, ds: str, tgt_split: str) -> str:
    """Draw line plots over p for RR and BR."""
    sns.set(style="whitegrid", font_scale=0.95)

    p_cats    = [str(p) for p in P_LIST]
    df["p_cat"] = pd.Categorical(df["p"].astype(str), categories=p_cats, ordered=True)

    metrics   = ["RR_tgt", "BR_all"]
    row_types = ["src_tgt", "tgt_fp", "tgt_fn"]

    base_size = 3
    fig, axes = plt.subplots(len(metrics), len(row_types),
                              figsize=(base_size * 4, base_size * 2))

    for i, metric in enumerate(metrics):
        for j, mt in enumerate(row_types):
            subset = df[df["misclf_type"] == mt]
            ax = axes[i][j]
            if subset.empty:
                ax.axis("off")
                continue
            sns.lineplot(
                data=subset,
                x="p_cat",
                y=metric,
                color="C0",
                marker="^",
                dashes=False,
                sort=False,
                ax=ax,
                mec="black",
                legend=False,
            )
            ax.set_title(f"{mt.upper().replace('_', '-')}", fontsize=12, fontstyle="italic")
            ax.set_xlabel("")
            if i == len(metrics) - 1:
                ax.set_xlabel("$p$")
            if j == 0:
                if metric == "RR_tgt":
                    ax.set_ylabel("Repair Rate")
                elif metric == "BR_all":
                    ax.set_ylabel("Break Rate")
            else:
                ax.set_ylabel("")
            ax.grid(True, linestyle="--", linewidth=0.5)

    handles = [
        Line2D([], [], marker="^", linestyle="", color="C0",
               label="REPTRAN", markersize=8, markeredgecolor="black")
    ]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.1), ncol=1, frameon=True, fontsize=12)
    fig.subplots_adjust(bottom=0.01, hspace=0.2, wspace=0.25)
    plt.tight_layout(rect=[0, 0.14, 1, 1])

    out_pdf = f"exp-repair-6-1_{ds}_{tgt_split}_p_lineplots.pdf"
    plt.savefig(out_pdf, dpi=150, bbox_inches="tight")
    print(f"[INFO] plot saved to {out_pdf}")
    return out_pdf


def run_kruskal_over_p(df: pd.DataFrame, ds: str, tgt_split: str) -> pd.DataFrame:
    """Run Kruskal-Wallis test over p for RR and BR."""
    results = []
    for metric_col, metric_label in [("RR_tgt", "RR"), ("BR_all", "BR")]:
        df_sub  = df[df["dataset"] == ds]
        grouped = df_sub.groupby("p")[metric_col].apply(list)
        if len(grouped) >= 2:
            try:
                h_val, p_val = kruskal(*grouped)
                n    = df_sub.shape[0]
                k    = len(grouped)
                eta2 = (h_val - k + 1) / (n - k) if n > k else float("nan")
            except Exception:
                h_val, p_val, eta2 = float("nan"), float("nan"), float("nan")
        else:
            h_val, p_val, eta2 = float("nan"), float("nan"), float("nan")
        results.append({
            "Dataset": ds, "Split": tgt_split, "Metric": metric_label,
            "H-statistic": h_val, "p-value": p_val, "η²": eta2,
        })
    return pd.DataFrame(results)


if __name__ == "__main__":
    ds_list       = ["c100", "tiny-imagenet"]
    tgt_split_list = ["test"]

    for ds_arg in ds_list:
        for tgt_split_arg in tgt_split_list:
            print(f"{'='*90}\nProcessing: dataset={ds_arg}, split={tgt_split_arg}")

            df_all = collect_records(ds_arg, tgt_split_arg)
            out_csv = f"exp-repair-6-1_{ds_arg}_{tgt_split_arg}_results_all.csv"
            df_all.to_csv(out_csv, index=False)
            print(f"[INFO] saved {out_csv}")

            plot_p_sensitivity(df_all, ds_arg, tgt_split_arg)

            kruskal_df = run_kruskal_over_p(df_all, ds_arg, tgt_split_arg)
            out_kruskal = f"exp-repair-6-1_{ds_arg}_{tgt_split_arg}_kruskal_p.csv"
            kruskal_df.to_csv(out_kruskal, index=False)
            print(f"[INFO] Kruskal results saved to {out_kruskal}")
            print(kruskal_df.to_string(index=False))
