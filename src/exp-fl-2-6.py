import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from itertools import product
from collections import defaultdict
from utils.constant import ViTExperiment
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")

def calc_eff_size_r(x, y):
    # 差がゼロでないペアの数をカウント
    diff = x - y
    valid_pairs = diff[diff != 0]
    n = len(valid_pairs)
    # 効果量の計算
    expected = n * (n + 1) / 4
    std = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z_value = (stat - expected) / std
    effect_size = abs(z_value) / np.sqrt(n)
    return effect_size

def calc_eff_size_a12(n1, n2, u_stat):
    """
    Calculate Vargha and Delaney's A12 statistic from U statistic.

    Parameters:
    - n1 (int): Sample size of group 1
    - n2 (int): Sample size of group 2
    - u_stat (float): Mann-Whitney U statistic

    Returns:
    - a12 (float): The Vargha and Delaney A12 statistic
    """
    if n1 == 0 or n2 == 0:
        raise ValueError("Sample sizes n1 and n2 must be greater than zero.")
    # print(f"Calculating A12 with n1={n1}, n2={n2}, u_stat={u_stat}")
    # Calculate A12 using the formula: A12 = U / (n1 * n2)
    a12 = u_stat / (n1 * n2)
    # print(f"Calculated A12: {a12}")
    return a12


if __name__ == "__main__":
    ds = "c100"
    true_labels = range(100) if ds == "c100" else None
    tgt_split = "repair"
    exp_fl_1_save_path = f"./exp-fl-1_{ds}_proba_diff.csv"
    exp_fl_2_save_path = f"./exp-fl-2_{ds}_proba_diff.csv"
    df1 = pd.read_csv(exp_fl_1_save_path)
    df2 = pd.read_csv(exp_fl_2_save_path)
    df = pd.concat([df1, df2], ignore_index=True)
    print(f"df.shape: {df.shape}")
    # df_run_allのdiff_proba_mean以外のユニークな値のリストを表示
    for col in df.columns:
        if col not in ["diff_proba"]:
            print(f"{col}: {df[col].unique()}")
    # ある列が特定の値の行のみ抽出
    df_extracted = df
    # df_extracted = df[(df["misclf_type"] == "src_tgt") & (df["tgt_rank"] == 1)]
    # label列の値, op列の値, fl_method列の値 ごとにdiff_probaの平均を表にする
    grouped_df = (
        df_extracted
        .groupby(["label", "op", "fl_method", "fl_target"])
        .agg(mean_diff_proba=("diff_proba", "mean"), std_diff_proba=("diff_proba", "std"))
        .reset_index()
    )
    # grouped_df = df_extracted.groupby(["label", "op", "fl_method", "fl_target"])["diff_proba"].mean().reset_index()
    grouped_df["mean_diff_proba"] = grouped_df["mean_diff_proba"] * 100
    print(f"grouped_df.shape: {grouped_df.shape}")
    
    # 検定結果を保存するリスト
    results = []
    # fl_methodのリスト
    fl_method_list = grouped_df["fl_method"].unique()
    
    for i, (fl_target, op) in enumerate(product(
        grouped_df["fl_target"].unique(), grouped_df["op"].unique()
    )):
        print(f"fl_target: {fl_target}, op: {op}")
        print("======================================")
        diff_proba_list = defaultdict(list)
        # fl_methodごとにdiff_probaのリストを作成
        for fl_method in fl_method_list:
            # fl_target, op, fl_method に対応するデータを取得
            subset = grouped_df[(grouped_df["op"] == op) & (grouped_df["fl_method"] == fl_method) & (grouped_df["fl_target"] == fl_target)]
            diff_proba_list[fl_method] = subset["mean_diff_proba"].to_list()

        # fl_method ごとの対応のある t 検定を実行
        fl_method_pairs = list(product(fl_method_list, repeat=2))
        for method_a, method_b in fl_method_pairs:
            if method_a >= method_b:  # 同じ組み合わせを繰り返さないように
                continue

            # method_a と method_b のデータを取得
            proba_a = np.array(diff_proba_list[method_a])
            proba_b = np.array(diff_proba_list[method_b])

            # データが空の場合はスキップ
            if len(proba_a) == 0 or len(proba_b) == 0:
                print(f"Skipping comparison: {method_a} vs {method_b} (empty data)")
                continue
            
            # 非パラメトリック検定
            if len(proba_a) == len(proba_b):  # ウィルコクソン検定
                print(f"Performing Wilcoxon signed-rank test for {method_a} and {method_b}")
                stat, p_value = wilcoxon(proba_a, proba_b, alternative='two-sided')
                # 効果量の計算
                eff_r = calc_eff_size_r(proba_a, proba_b)
                eff_a12 = calc_eff_size_a12(len(proba_a), len(proba_b), stat)
            else:
                print(f"Skipping comparison: {method_a} vs {method_b} (non-normal data and unequal lengths)")
                continue
            
            
            # 検定結果をリストに追加
            results.append({
                "fl_target": fl_target,
                "op": op,
                "method_a": method_a,
                "method_b": method_b,
                "stat": stat,
                "p_value": p_value,
                "effect_size_r": eff_r,
                "effect_size_a12": eff_a12
            })
            
            # 検定結果を表示
            print(f"{fl_target}, {op}: {method_a} vs {method_b} -> t_stat: {stat}, p_value: {p_value}")

    # 検定結果を DataFrame に変換して保存
    results_df = pd.DataFrame(results)
    # p_valueの調整
    results_df["p_value_corrected"] = multipletests(results_df["p_value"], method="fdr_bh")[1]
    # 棄却されたかどうかのフラグ
    results_df["reject_null"] = multipletests(results_df["p_value"], method="fdr_bh")[0]
    results_save_path = f"./exp-fl-2_{ds}_wilcoxon_single_rank_test_results.csv"
    results_df.to_csv(results_save_path, index=False)
    print(f"Paired Wilcoxon signed-rank test results saved to {results_save_path}")