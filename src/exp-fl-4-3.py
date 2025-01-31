import os
import numpy as np
import pandas as pd
from utils.constant import ViTExperiment
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")

if __name__ == "__main__":
    ds = "c100"
    true_labels = range(100) if ds == "c100" else None
    tgt_split = "repair"
    exp_fl_1_save_path = f"./exp-fl-1_{ds}_proba_diff.csv"
    exp_fl_2_save_path = f"./exp-fl-2_{ds}_proba_diff.csv"
    exp_fl_3_save_path = f"./exp-fl-3_{ds}_proba_diff.csv"

    df1 = pd.read_csv(exp_fl_1_save_path)
    df2 = pd.read_csv(exp_fl_2_save_path)
    df3 = pd.read_csv(exp_fl_3_save_path)
    print(f"df1.shape: {df1.shape}")
    print(f"df2.shape: {df2.shape}")
    print(f"df3.shape: {df3.shape}")

    # df1,2,3を縦に結合
    ori_df = pd.concat([df1, df2, df3], ignore_index=True)
    print(f"df.shape: {ori_df.shape}")

    # num_weight のユニークな値ごとに処理
    for wnum in ori_df["num_weight"].unique():
        df = ori_df[ori_df["num_weight"] == wnum]

        # (fl_method, fl_target) の行Index
        method_target_pairs = (
            df[["fl_target", "fl_method"]]
            .drop_duplicates()
            .sort_values(["fl_target", "fl_method"])
            .apply(tuple, axis=1)
            .tolist()
        )
        row_index = pd.MultiIndex.from_tuples(
            method_target_pairs, names=["fl_target", "fl_method"]
        )

        # 集約結果を格納するテーブル
        result_table = pd.DataFrame(index=row_index, columns=["total_misclf"], data=0)

        # `groupby` で `mean_diff_proba` を計算
        grouped_df = (
            df
            .groupby(["label", "op", "fl_method", "fl_target"], as_index=False)
            .agg(
                count_rows=("diff_proba", "count"),
                mean_diff_proba=("diff_proba", "mean"),
                std_diff_proba=("diff_proba", "std"))
        )
        grouped_df["mean_diff_proba"] = grouped_df["mean_diff_proba"] * 100
        print(grouped_df)

        # fl_method, fl_target ごとに「ラベルごと最大の行」を取得し、>0 のクラスをカウント
        for (method_val, target_val), sub_df in grouped_df.groupby(["fl_method", "fl_target"]):
            print(f"\nmethod: {method_val}, target: {target_val}")
            print(sub_df)
            if len(sub_df) == 0:
                continue

            sub_max = sub_df.groupby("label").apply(
                lambda g: g.loc[g["mean_diff_proba"].idxmax()]
            )
            print(sub_max)
            n_pos = (sub_max["mean_diff_proba"] > 0).sum()
            print(n_pos)

            # `total_misclf` に書き込み
            result_table.loc[(target_val, method_val), "total_misclf"] = n_pos

        # CSVに保存
        out_csv = f"./exp-fl-4-3_{ds}_no_rank_table_wnum{wnum}.csv"
        desired_index_order = [
            ("neuron", "random"),
            ("weight", "random"),
            ("neuron", "ig"),
            ("weight", "bl"),
            ("neuron", "vdiff"),
            ("weight", "vdiff"),
            ("neuron", "vdiff+mean_act"),
            ("weight", "vdiff+mean_act+grad"),
        ]
        result_table = result_table.reindex(desired_index_order)
        result_table.to_csv(out_csv)
        print(f"Saved table to {out_csv}")
