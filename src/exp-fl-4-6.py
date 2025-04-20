import os
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import pandas as pd
from utils.helper import get_device, json2dict
from utils.constant import ViTExperiment, ExperimentRepair1, Experiment3, ExperimentRepair2, Experiment4
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

device = get_device()

def get_loss_diff_path(n, beta, method, loss_diff_dir, op, cor_mis):
    assert cor_mis in ["cor", "mis"], f"Unknown cor_mis: {cor_mis}"
    if method == "ours":
        return os.path.join(loss_diff_dir, f"exp-fl-6_loss_diff_n{n}_beta{beta}_{op}_{cor_mis}_weight_ours.npy")
    elif method == "bl":
        return os.path.join(loss_diff_dir, f"exp-fl-2_loss_diff_n{n}_{op}_{cor_mis}_weight_bl.npy")
    elif method == "random":
        return os.path.join(loss_diff_dir, f"exp-fl-1_loss_diff_n{n}_{op}_{cor_mis}_weight_random.npy")
    else:
        raise ValueError(f"Unknown method: {method}")


def load_best_loss_diff(n, beta, method, tgt_rank, misclf_type, sample_type, ops=None):
    if ops is None:
        ops = ["enhance", "suppress"]
    loss_diff_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_weights_location", "loss_diff_per_sample")
    paths = {
        op: get_loss_diff_path(n, beta, method, loss_diff_dir, op, sample_type)
        for op in ops
    }
    # 存在チェック & 読み込み
    losses = {}
    for op, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"'{op}' の loss_diff ファイルが見つかりません: {path}")
        losses[op] = np.load(path)
    # 形状チェック
    shapes = [arr.shape for arr in losses.values()]
    if not all(shape == shapes[0] for shape in shapes):
        raise AssertionError(f"Shape mismatch among operations: {dict(zip(ops, shapes))}")
    losses = {op: np.load(path) for op, path in paths.items() if os.path.exists(path)}
    # best
    # shape = (num_ops, ...) -> 各サンプルで最小値を取る
    stacked = np.stack(list(losses.values()), axis=0)
    best_losses = np.min(stacked, axis=0)

    return best_losses

def cliffs_delta_from_diff(diff):
    gt = np.sum(diff > 0)
    lt = np.sum(diff < 0)
    n = len(diff)
    return (gt - lt) / n

if __name__ == "__main__":
    # 比較対象とパラメータ
    methods = ["ours", "bl", "random"]
    beta_list = [0.1, 0.25, 0.5, 0.75, 1.0]
    comparison_pairs = defaultdict(list)
    for beta in beta_list:
        comparison_pairs[beta] = [(("ours", beta), "bl"), (("ours", beta), "random"), ("bl", "random")]
    tgt_ranks = [1, 2, 3, 4, 5]
    misclf_types = ["src_tgt", "tgt", "tgt_fp", "tgt_fn"]
    sample_types = ["cor", "mis"]
    pretrained_dir = ViTExperiment.c100.OUTPUT_DIR.format(k=0) # /src/out_vit_c100_fold0
    # n_list = [ExperimentRepair2.NUM_IDENTIFIED_WEIGHTS, ExperimentRepair1.NUM_IDENTIFIED_WEIGHTS, Experiment3.NUM_IDENTIFIED_WEIGHTS]
    n_list = [Experiment4.NUM_IDENTIFIED_WEIGHTS]
    op_list = ["enhance", "suppress", "multiply-2"]
    
    # =========================================================
    # 1.  ハイパーパラメータ
    # =========================================================
    alpha             = 0.05
    correction_method = "bonferroni"   # "holm", "fdr_bh", …
    thr_delta         = 0.474          # |Δ| ≥ 0.474 : Cliff’s delta “large”

    # =========================================================
    # 2.  判定ヘルパ
    # =========================================================
    def is_sig(p):               return p < alpha
    def large_pos(d):            return d >  thr_delta
    def large_neg(d):            return d < -thr_delta

    # =========================================================
    # 3.  解析ループ
    # =========================================================
    for n in n_list:
        count_records = []                     # ← ここへ集計を追加
        for beta, comp_pair in comparison_pairs.items():        # ← 既存の比較ペア辞書
            print(f"n = {n}, β = {beta}")
            fig, axes = plt.subplots(2, len(comp_pair), figsize=(24,16))
            cmap = "RdBu_r"

            for col_idx, (mA_raw, mB_raw) in enumerate(comp_pair):

                method_a = mA_raw[0] if isinstance(mA_raw, tuple) else mA_raw
                method_b = mB_raw[0] if isinstance(mB_raw, tuple) else mB_raw
                ax_dic   = {"cor": axes[0, col_idx], "mis": axes[1, col_idx]}

                for st in sample_types:

                    p_list, delta_list, idx_list = [], [], []

                    # -------- Wilcoxon & Δ 計算 --------
                    for i, r in enumerate(tgt_ranks):
                        for j, mc in enumerate(misclf_types):
                            a = load_best_loss_diff(n, beta, method_a, r, mc, st, ops=op_list)
                            b = load_best_loss_diff(n, beta, method_b, r, mc, st, ops=op_list)

                            _, p   = wilcoxon(a, b)
                            delta  = cliffs_delta_from_diff(a - b)

                            p_list.append(p)
                            delta_list.append(delta)
                            idx_list.append((i,j))

                    # -------- 多重補正 --------
                    reject, p_corr_all, *_ = multipletests(p_list, alpha=alpha,
                                                        method=correction_method)

                    # -------- 表示用行列 & カウント --------
                    mat       = np.zeros((len(tgt_ranks), len(misclf_types)))
                    annot     = np.empty_like(mat, dtype=object)
                    pos_cnt   = np.zeros(len(misclf_types), dtype=int)   # 正方向 large & 有意
                    neg_cnt   = np.zeros(len(misclf_types), dtype=int)   # 負方向 large & 有意

                    for (i,j), p_corr, rej, delta in zip(idx_list, p_corr_all, reject, delta_list):
                        mat[i,j] = delta
                        star     = "**" if (rej and p_corr < 0.01) else "*" if (rej and p_corr < 0.05) else ""
                        annot[i,j] = f"{delta:+.3f}{star}"

                        # ---- large & 有意 を方向別にカウント ----
                        if is_sig(p_corr) and large_pos(delta): pos_cnt[j] += 1
                        if is_sig(p_corr) and large_neg(delta): neg_cnt[j] += 1

                    # -------- ヒートマップ --------
                    sns.heatmap(pd.DataFrame(mat, index=tgt_ranks, columns=misclf_types),
                                ax=ax_dic[st], annot=annot, fmt="", cmap=cmap,
                                center=0, vmin=-1, vmax=1,
                                cbar_kws={"shrink":0.6}, annot_kws={"size":11})
                    title = f"{method_a} vs. {method_b}".replace("ours", f"ours(β={beta})")
                    ax_dic[st].set_title(f"{title} – {st}", fontsize=14)
                    ax_dic[st].set_xlabel("") ; ax_dic[st].set_ylabel("tgt_rank")

                    # -------- 集計を１行にまとめて保存 --------
                    record = {
                        "beta"   : beta,
                        "pair"   : f"{method_a}–{method_b}",
                        "sample" : st,
                    }
                    # (正カウント , 負カウント) のタプルを列に
                    for j, mc in enumerate(misclf_types):
                        record[mc] = (int(pos_cnt[j]), int(neg_cnt[j]))
                    count_records.append(record)

            # plt.tight_layout(); plt.show()

        # =========================================================
        # 4.  DataFrame 整形 → Multi‑Index 列
        # =========================================================
        count_df = pd.DataFrame(count_records)

        # ▷ “縦”→“横” : cor / mis を列 Multi‑Index 上位レベルに
        wide = (
            count_df
            .set_index(["beta","pair","sample"])       # 行：β × ペア × sample
            [misclf_types]                            # ← tuple を持つ４列
            .unstack("sample")                        # 列：sample(level‑0) × misclf_type(level‑1)
        )
        
        # ─────────── ここから追加 ───────────
        # 1) フラット化
        df = wide.reset_index()

        # 2) bl–random を一行だけ抽出して beta="-" に
        blr = df[df["pair"]=="bl–random"].iloc[[0]].copy()
        blr["beta"] = "-"

        # 3) 残り (ours–bl, ours–random) を pair→beta でソート
        others = df[df["pair"]!="bl–random"].copy()
        # pair をカテゴリ順に
        others["pair"] = pd.Categorical(
            others["pair"],
            categories=["ours–bl", "ours–random"],
            ordered=True
        )
        others = others.sort_values(["pair","beta"], ascending=[True,True])

        # 4) 再結合して MultiIndex に戻す
        final = pd.concat([others, blr], ignore_index=True)
        final = final.set_index(["beta","pair"])
        wide = final
        # ─────────── ここまで ───────────

        # =========================================================
        # 5.  CSV 保存（必要なら）
        # =========================================================
        wide.to_csv(f"exp-fl-4-6_n{n}.csv")
        print(f"Saved to exp-fl-4-6_n{n}.csv")
