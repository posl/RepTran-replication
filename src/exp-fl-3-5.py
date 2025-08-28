import os
import numpy as np
import pandas as pd
from itertools import product
from collections import defaultdict
from utils.constant import ViTExperiment
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
sns.set_style("ticks")

if __name__ == "__main__":
    # only_suppressというバイナリ変数をargparseで受けとる
    parser = argparse.ArgumentParser()
    parser.add_argument("--only_suppress", action="store_true", default=False)
    args = parser.parse_args()
    only_suppress = args.only_suppress
    
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
    df = pd.concat([df1, df2, df3])
    # Display list of unique values other than diff_proba_mean in df_run_all
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
    print(f"grouped_df.shape: {grouped_df.shape}")
    
    for target in grouped_df["fl_target"].unique():
        filename = f"./exp-fl-3-5_{ds}_proba_diff_{target}"
        # grouped_df のうち fl_target 列が現在の target の値に一致する行を特定
        filtered_grouped_df = grouped_df[grouped_df["fl_target"] == target]
        
        # FL-methodごとの箱ひげ図を作成する
        plt.figure(figsize=(10, 6))
        boxplot_data = [
            filtered_grouped_df[filtered_grouped_df["fl_method"] == method]["mean_diff_proba"]
            for method in filtered_grouped_df["fl_method"].unique()
        ]
        plt.boxplot(
            boxplot_data,
            labels=filtered_grouped_df["fl_method"].unique(),
        )
        plt.xlabel("FL-method")
        plt.ylabel("Average Change of Probability (%)")
        plt.title(f"({target}-level) Distribution of Probability Changes by FL-method")
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(filename + "_box.pdf", dpi=300, bbox_inches="tight")
        plt.savefig(filename + "_box.png", dpi=300, bbox_inches="tight")
        
        # FL-methodごとのKDEプロットを作成する
        plt.figure(figsize=(10, 6))
        for method in filtered_grouped_df["fl_method"].unique():
            sns.kdeplot(
                filtered_grouped_df[filtered_grouped_df["fl_method"] == method]["mean_diff_proba"],
                label=method,
                fill=True,
            )
        # x=0に縦線を引く
        plt.axvline(x=0, color='gray', linestyle='--')
        plt.xlabel("Average Change of Probability (%)")
        plt.ylabel("Density")
        plt.title(f"({target}-level) Distribution of Probability Changes by FL-method")
        plt.legend(loc='upper left')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(filename + "_kde.pdf", dpi=300, bbox_inches="tight")
        plt.savefig(filename + "_kde.png", dpi=300, bbox_inches="tight")