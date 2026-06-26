"""
exp-repair-7-1-5.py  —  Meta3 / RQ6: integrate 4 conditions, plot, and run stats.

Equal-budget (N_w=472) ablation of the neuron score.  Four conditions:
    Full            : VDiff x MisAct  (= Ours, reused from exp-repair-4-1, fl_method=ours)
    VDiff-only      : VDiff           (new, exp-repair-7, score_mode=vdiff)
    MisAct-only     : MisAct          (new, exp-repair-7, score_mode=misact)
    No-neuron-score : uniform 1       (= ARACHNEW, reused from exp-repair-4-1, fl_method=bl)

For each dataset we collect RR (repair_rate_tgt) and BR (break_rate_overall) over the
9 misclassification benchmarks (3 ranks x {src_tgt, tgt_fp, tgt_fn}) x 5 reps, then:
  - dump the raw 4 x 9 x 5 data                        -> exp-repair-7-1_{ds}_test_results_all.csv
  - draw per-benchmark + overall box plots (RR, BR)    -> exp-repair-7-1_{ds}_ablation_plots.pdf
  - paired Wilcoxon signed-rank + Cliff's delta for    -> exp-repair-7-1_{ds}_test_stats.csv
        Full vs VDiff-only / Full vs MisAct-only /
        Full vs No-neuron-score / VDiff-only vs MisAct-only
"""
import os, json, warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from utils.constant import ViTExperiment


def holm_correct(p_raw):
    """Holm-Bonferroni step-down adjusted p-values (drop-in replacement for
    statsmodels.stats.multitest.multipletests(..., method="holm")[1], so no
    statsmodels dependency is required). NaN inputs (e.g. Wilcoxon on all-zero
    differences) are treated as 1.0."""
    p = np.asarray(p_raw, dtype=float)
    p = np.where(np.isnan(p), 1.0, p)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p)
    p_sorted = p[order]
    adj_sorted = (m - np.arange(m)) * p_sorted      # step-down scaling
    adj_sorted = np.maximum.accumulate(adj_sorted)  # enforce monotonicity
    adj_sorted = np.clip(adj_sorted, None, 1.0)
    p_adj = np.empty(m, dtype=float)
    p_adj[order] = adj_sorted
    return p_adj

NUM_REPS         = 5
FIXED_WNUM       = 472
ALPHA_IN_ARACHNE = 10
ALPHA_STR        = ALPHA_IN_ARACHNE / (1 + ALPHA_IN_ARACHNE)   # -> '0.9090909090909091'
FIXED_BOUNDS     = "Arachne"
TGT_SPLIT        = "test"

K_LIST        = [0]
TGT_RANK_LIST = [1, 2, 3]
MISCLF_LIST   = ["src_tgt", "tgt_fp", "tgt_fn"]

# Condition -> (origin, short label).  Origin tells which JSON file to read.
CONDITIONS = ["Full", "VDiff", "MisAct", "None"]
COND_LABEL = {"Full": "Full", "VDiff": "VDiff-only",
              "MisAct": "MisAct-only", "None": "No-neuron-score"}

# first element = DataFrame column name produced by collect_records (NOT the raw
# JSON key, which is read directly via js.get() there); used by plot_ablation & run_stats.
METRIC_INFO = dict(
    RR=("RR_tgt", "Repair Rate"),
    BR=("BR_all", "Break Rate"),
)

# Paired contrasts (md: 4 contrasts)
PAIRS = [("Full", "VDiff"),
         ("Full", "MisAct"),
         ("Full", "None"),
         ("VDiff", "MisAct")]


def metrics_path(condition, save_dir, reps):
    """Return the test-metrics JSON path for a given condition."""
    if condition == "Full":      # reuse exp-repair-4-1 ours
        base = f"n{FIXED_WNUM}_alpha{ALPHA_STR}_boundsArachne_ours"
        return os.path.join(save_dir, f"exp-repair-4-1-metrics_for_{TGT_SPLIT}_{base}_reps{reps}.json")
    if condition == "None":      # reuse exp-repair-4-1 bl (ARACHNEW)
        base = f"n{FIXED_WNUM}_alpha{ALPHA_STR}_boundsArachne_bl"
        return os.path.join(save_dir, f"exp-repair-4-1-metrics_for_{TGT_SPLIT}_{base}_reps{reps}.json")
    # new exp-repair-7 conditions
    score_mode = "vdiff" if condition == "VDiff" else "misact"
    base = f"n{FIXED_WNUM}_alpha{ALPHA_STR}_boundsArachne_{score_mode}_ours"
    return os.path.join(save_dir, f"exp-repair-7-1-metrics_for_{TGT_SPLIT}_{base}_reps{reps}.json")


def collect_records(ds: str) -> pd.DataFrame:
    """Scan JSON results and return the long-form 4 x 9 x 5 DataFrame for a dataset."""
    records = []
    for k in K_LIST:
        pretrained_dir = getattr(ViTExperiment, ds.replace("-", "_")).OUTPUT_DIR.format(k=k)
        for tgt_rank in TGT_RANK_LIST:
            for misclf_full in MISCLF_LIST:
                save_dir = os.path.join(
                    pretrained_dir, f"misclf_top{tgt_rank}",
                    f"{misclf_full}_repair_weight_by_de",
                )
                for condition in CONDITIONS:
                    for reps in range(NUM_REPS):
                        jp = metrics_path(condition, save_dir, reps)
                        if not os.path.exists(jp):
                            raise FileNotFoundError(jp)
                        with open(jp) as f:
                            js = json.load(f)
                        records.append({
                            "dataset":     ds,
                            "condition":   condition,
                            "cond_label":  COND_LABEL[condition],
                            "misclf_type": misclf_full,
                            "tgt_rank":    tgt_rank,
                            "reps_id":     reps,
                            "benchmark":   f"{misclf_full}_r{tgt_rank}",
                            "RR_tgt":      js.get("repair_rate_tgt"),
                            "BR_all":      js.get("break_rate_overall"),
                            "delta_acc":   js.get("delta_acc"),
                        })
    return pd.DataFrame(records)


def paired_cliffs_delta(v1: np.ndarray, v2: np.ndarray) -> float:
    """Paired Cliff's delta: (n_pos - n_neg) / N over per-pair differences."""
    diff = v1 - v2
    return (np.sum(diff > 0) - np.sum(diff < 0)) / diff.size


def run_stats(df: pd.DataFrame, ds: str) -> pd.DataFrame:
    """Paired Wilcoxon + Cliff's delta for the 4 contrasts x {RR, BR}.

    Pairing is per (benchmark, reps) point; the 4 contrasts share Holm correction
    within each metric.
    """
    rows = []
    for met_tag, (col, _nice) in METRIC_INFO.items():
        # Build a wide matrix indexed by (benchmark, reps) so points align across conditions.
        wide = (df.pivot_table(index=["benchmark", "reps_id"],
                               columns="condition", values=col)
                  .sort_index())
        p_raw, recs = [], []
        for m1, m2 in PAIRS:
            v1 = wide[m1].to_numpy()
            v2 = wide[m2].to_numpy()
            mask = ~(np.isnan(v1) | np.isnan(v2))
            v1, v2 = v1[mask], v2[mask]
            if np.allclose(v1, v2):
                p = 1.0
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    p = wilcoxon(v1, v2, zero_method="wilcox").pvalue
            d = paired_cliffs_delta(v1, v2)
            p_raw.append(p)
            recs.append({
                "dataset":   ds,
                "metric":    met_tag,
                "contrast":  f"{COND_LABEL[m1]} vs {COND_LABEL[m2]}",
                "n_pairs":   int(v1.size),
                "cliffs_delta": d,
                "p_raw":     p,
            })
        p_adj = holm_correct(p_raw)
        for rec, pa in zip(recs, p_adj):
            rec["p_holm"] = pa
            rows.append(rec)
    return pd.DataFrame(rows)


def plot_ablation(df: pd.DataFrame, ds: str) -> str:
    """Per-benchmark + overall box plots of RR and BR for the 4 conditions."""
    sns.set(style="whitegrid", font_scale=0.85)
    hue_order = [COND_LABEL[c] for c in CONDITIONS]

    # benchmark order: keep grouped by misclf_type then rank, then an "overall" panel
    bench_order = [f"{m}_r{r}" for m in MISCLF_LIST for r in TGT_RANK_LIST]

    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    for ax, (met_tag, (col, nice)) in zip(axes, METRIC_INFO.items()):
        sns.boxplot(
            data=df, x="benchmark", y=col, hue="cond_label",
            order=bench_order, hue_order=hue_order, ax=ax,
            showfliers=False, palette="Set2",
        )
        ax.set_title(f"{nice} per benchmark (N_w={FIXED_WNUM}, equal-budget)", fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel(nice)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, linestyle="--", linewidth=0.5)
        if ax is axes[0]:
            ax.legend(title="", loc="upper right", ncol=4, fontsize=9)
        else:
            ax.get_legend().remove()

    fig.suptitle(f"Neuron-score component ablation — {ds}", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out_pdf = f"exp-repair-7-1_{ds}_ablation_plots.pdf"
    plt.savefig(out_pdf, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] plot saved to {out_pdf}")
    return out_pdf


if __name__ == "__main__":
    ds_list = ["c100", "tiny-imagenet"]

    for ds_arg in ds_list:
        print(f"{'='*90}\nProcessing: dataset={ds_arg}")

        df_all = collect_records(ds_arg)
        out_csv = f"exp-repair-7-1_{ds_arg}_{TGT_SPLIT}_results_all.csv"
        df_all.to_csv(out_csv, index=False)
        print(f"[INFO] saved {out_csv}  ({len(df_all)} rows)")

        # quick mean summary to stdout
        summary = (df_all.groupby("cond_label")[["RR_tgt", "BR_all"]]
                   .mean().reindex([COND_LABEL[c] for c in CONDITIONS]))
        print(summary.to_string())

        plot_ablation(df_all, ds_arg)

        stats_df = run_stats(df_all, ds_arg)
        out_stats = f"exp-repair-7-1_{ds_arg}_{TGT_SPLIT}_stats.csv"
        stats_df.to_csv(out_stats, index=False)
        print(f"[INFO] stats saved to {out_stats}")
        print(stats_df.to_string(index=False))
