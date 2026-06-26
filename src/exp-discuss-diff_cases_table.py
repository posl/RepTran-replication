"""exp-discuss-diff_cases_table.py

Tabular companion to exp-discuss-diff_cases.ipynb (the 3-set Venn, Discussion).
It reports, *as a table*, the same "method-exclusive" information the Venn shows
in its unique regions (100/010/001) — but as counts instead of circles.

For each of the 18 benchmarks
    {C100, TinyImg} x rank{1,2,3} x {SRC-TGT, TGT-FP, TGT-FN}
and each weight budget N_w in {236, 472, 944}, we count, per method
(REPTRAN / ArachneW / PRoViT_{FT+LP}):

    uRepair : test instances UNIQUELY repaired by that method, consistently
              across ALL 5 runs (5/5)   -> repair_indices_tgt
    uBreak  : test instances UNIQUELY broken  by that method, consistently
              across ALL 5 runs (5/5)   -> break_indices_overall

"unique" = exclusive to that method among the three (= the Venn's 100/010/001).
REPTRAN/ArachneW/PRoViT use the SAME index spaces (cf. the notebook), so the
set differences are well-defined. PRoViT (FT+LP) has a single configuration and
is independent of N_w; its sets are the same across the three tables, while its
*unique* counts can still differ (REPTRAN/ArachneW sets change with N_w).

Outputs (written next to this script, in src/):
    exp-discuss-diff_cases_unique_table_n{Nw}.csv   (one 18x6 table per N_w)
    exp-discuss-diff_cases_unique_table_all.csv     (all three stacked)
    exp-discuss-diff_cases_unique_table.tex         (LaTeX, booktabs, per N_w)

Usage (inside the docker container):
    python exp-discuss-diff_cases_table.py
"""
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from utils.constant import ViTExperiment

# ---- config (mirrors exp-discuss-diff_cases.ipynb) -------------------------
DS_LIST       = ["c100", "tiny-imagenet"]
K             = 0
TGT_RANK_LIST = [1, 2, 3]
BENCH_TFP     = [("src_tgt", None), ("tgt", "fp"), ("tgt", "fn")]
W_NUM_LIST    = [236, 472, 944]
TGT_SPLIT     = "test"
NUM_REPS      = 5
ALPHA_STR     = "0.9090909090909091"

PROVIT_SUBDIR = "provit_ft_lp_rerun"
PROVIT_LR     = 0.001
PROVIT_EPS    = 0.01

# repaired & broken both require consistency across ALL runs (5/5),
# matching the Venn figures (consistent_set uses c == num_reps).
REPAIR_THRESHOLD = NUM_REPS   # 5/5
BREAK_THRESHOLD  = NUM_REPS   # 5/5


# ---- loaders (same file paths / index spaces as the notebook) -------------
def load_indices(base, rank, wnum, fl_method, rep, misclf, fpfn, itype):
    """REPTRAN (fl_method='ours') / ArachneW (fl_method='bl')."""
    setting_id = f"n{wnum}_alpha{ALPHA_STR}_boundsArachne"
    if misclf == "src_tgt":
        folder = f"misclf_top{rank}/{misclf}_repair_weight_by_de"
    elif misclf == "tgt" and fpfn is not None:
        folder = f"misclf_top{rank}/{misclf}_{fpfn}_repair_weight_by_de"
    else:
        return None
    fp = os.path.join(
        base, folder,
        f"exp-repair-4-1-change_indices_{TGT_SPLIT}_{setting_id}_{fl_method}_reps{rep}.npz")
    if not os.path.exists(fp):
        return None
    return set(np.load(fp)[itype].tolist())


def provit_bname(misclf, fpfn, rank):
    return f"{misclf}_{fpfn}_rank{rank}" if fpfn else f"{misclf}_rank{rank}"


def load_provit(base, rank, misclf, fpfn, rep, itype):
    """PRoViT (FT+LP) re-run, independent of N_w."""
    fp = os.path.join(
        base, PROVIT_SUBDIR, "checkpoints",
        f"change_indices_test_lr{PROVIT_LR}_eps{PROVIT_EPS}_"
        f"{provit_bname(misclf, fpfn, rank)}_rep{rep}.npz")
    if not os.path.exists(fp):
        return None
    return set(np.load(fp)[itype].tolist())


def load_denoms(base, rank, misclf, fpfn):
    """(|I_test_mis|, |I_test_cor|) for this benchmark, read from the PRoViT npz
    (which stores both). They are benchmark constants shared by every method and
    every N_w, and are used as the percentage denominators:
      repaired metrics over |I_test_mis| (the target-misclassified set),
      broken   metrics over |I_test_cor| (the originally-correct set)."""
    for rep in range(NUM_REPS):
        fp = os.path.join(
            base, PROVIT_SUBDIR, "checkpoints",
            f"change_indices_test_lr{PROVIT_LR}_eps{PROVIT_EPS}_"
            f"{provit_bname(misclf, fpfn, rank)}_rep{rep}.npz")
        if os.path.exists(fp):
            d = np.load(fp)
            return int(len(d["I_test_mis"])), int(len(d["I_test_cor"]))
    return None, None


def aggregate(loader, threshold):
    """Indices occurring in >= threshold of the NUM_REPS runs."""
    cnt = defaultdict(int)
    for rep in range(NUM_REPS):
        s = loader(rep)
        if s is None:
            continue
        for idx in s:
            cnt[idx] += 1
    return {idx for idx, c in cnt.items() if c >= threshold}


def method_sets(base, rank, misclf, fpfn, wnum, itype, threshold):
    ours = aggregate(lambda r: load_indices(base, rank, wnum, "ours", r, misclf, fpfn, itype),
                     threshold)
    bl   = aggregate(lambda r: load_indices(base, rank, wnum, "bl",   r, misclf, fpfn, itype),
                     threshold)
    prov = aggregate(lambda r: load_provit(base, rank, misclf, fpfn, r, itype),
                     threshold)
    return ours, bl, prov


def unique_counts(ours, bl, prov):
    """Counts exclusive to each method (Venn regions 100/010/001)."""
    return (len(ours - bl - prov),
            len(bl - ours - prov),
            len(prov - ours - bl))


def common_count(ours, bl, prov):
    """Count shared by ALL three methods (Venn region 111)."""
    return len(ours & bl & prov)


# ---- build the 18 x (N_w) x 6 table ---------------------------------------
def build_table():
    rows = []
    for ds in DS_LIST:
        base = getattr(ViTExperiment, ds.replace("-", "_")).OUTPUT_DIR.format(k=K)
        ds_repr = "C100" if ds == "c100" else "TinyImg"
        for rank in TGT_RANK_LIST:
            for misclf, fpfn in BENCH_TFP:
                bench = "SRC-TGT" if misclf == "src_tgt" else f"TGT-{fpfn.upper()}"
                denom_repair, denom_break = load_denoms(base, rank, misclf, fpfn)
                for wnum in W_NUM_LIST:
                    ro, rb, rp = method_sets(base, rank, misclf, fpfn, wnum,
                                             "repair_indices_tgt", REPAIR_THRESHOLD)
                    urep = unique_counts(ro, rb, rp)
                    bo, bb, bp = method_sets(base, rank, misclf, fpfn, wnum,
                                             "break_indices_overall", BREAK_THRESHOLD)
                    ubrk = unique_counts(bo, bb, bp)
                    rows.append({
                        "N_w": wnum,
                        "dataset": ds_repr,
                        "rank": rank,
                        "benchmark": bench,
                        "uRepair_REPTRAN":  urep[0],
                        "uRepair_ArachneW": urep[1],
                        "uRepair_PRoViT":   urep[2],
                        "uBreak_REPTRAN":   ubrk[0],
                        "uBreak_ArachneW":  ubrk[1],
                        "uBreak_PRoViT":    ubrk[2],
                        "denom_repair":     denom_repair,
                        "denom_break":      denom_break,
                    })
    return pd.DataFrame(rows)


REPAIR_COLS = ["uRepair_REPTRAN", "uRepair_ArachneW", "uRepair_PRoViT"]
BREAK_COLS  = ["uBreak_REPTRAN", "uBreak_ArachneW", "uBreak_PRoViT"]
VALUE_COLS  = REPAIR_COLS + BREAK_COLS
ID_COLS     = ["dataset", "rank", "benchmark"]


def _fmt(count, denom):
    """Plain count (percentages dropped for readability; denom kept in the
    combined CSV for anyone who wants to recompute ratios)."""
    return f"{int(count)}"


def slash_bold_max(vals):
    """'a/b/c' for LaTeX, bolding the maximum value(s) in the triple via
    \\textbf{}. If all three are equal, nothing is bolded."""
    vals = [int(v) for v in vals]
    mx = max(vals)
    all_equal = len(set(vals)) == 1
    parts = [(r"\textbf{%d}" % v) if (v == mx and not all_equal) else str(v)
             for v in vals]
    return "/".join(parts)


def formatted_view(sub):
    """ID columns + each value as 'count (pct%)' (repair vs break denominators)."""
    out = sub[ID_COLS].copy()
    for col in REPAIR_COLS:
        out[col] = [_fmt(c, d) for c, d in zip(sub[col], sub["denom_repair"])]
    for col in BREAK_COLS:
        out[col] = [_fmt(c, d) for c, d in zip(sub[col], sub["denom_break"])]
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    df = build_table()

    # one 18x6 counts table per N_w (CSV + console)
    for wnum in W_NUM_LIST:
        sub = df[df["N_w"] == wnum].reset_index(drop=True)
        view = formatted_view(sub)
        out = os.path.join(here, f"exp-discuss-diff_cases_unique_table_n{wnum}.csv")
        view.to_csv(out, index=False)
        print(f"\n===== N_w = {wnum}  (repaired 5/5 | broken 5/5; counts) =====")
        print(view.to_string(index=False))
        print(f"[saved] {out}")

    # combined CSV: raw counts + denominators (for any further analysis)
    all_out = os.path.join(here, "exp-discuss-diff_cases_unique_table_all.csv")
    df[["N_w"] + ID_COLS + VALUE_COLS + ["denom_repair", "denom_break"]].to_csv(all_out, index=False)
    print(f"\n[saved] {all_out}")

    # LaTeX (booktabs), one tabular per N_w.
    # Each metric is a single slash-separated cell: REPTRAN/ArachneW/PRoViT.
    tex_path = os.path.join(here, "exp-discuss-diff_cases_unique_table.tex")
    header = (" & ".join(["Dataset", "Rank", "Bench",
                          "Repaired (5/5)", "Broken (5/5)"]))
    subhdr = " & & & REPTRAN/ArachneW/PRoViT & REPTRAN/ArachneW/PRoViT"
    with open(tex_path, "w") as f:
        for wnum in W_NUM_LIST:
            sub = df[df["N_w"] == wnum].reset_index(drop=True)
            f.write(f"% N_w = {wnum}\n")
            f.write("\\begin{tabular}{lll cc}\n\\toprule\n")
            f.write(header + " \\\\\n")
            f.write(subhdr + " \\\\\n\\midrule\n")
            for _, r in sub.iterrows():
                rep = slash_bold_max([r[c] for c in REPAIR_COLS])
                brk = slash_bold_max([r[c] for c in BREAK_COLS])
                cells = [str(r["dataset"]), str(r["rank"]), str(r["benchmark"]), rep, brk]
                f.write(" & ".join(cells) + " \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n\n")
    print(f"[saved] {tex_path}")


if __name__ == "__main__":
    main()
