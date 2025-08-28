import os
import numpy as np
import pandas as pd
from itertools import product
from collections import defaultdict
from utils.constant import ViTExperiment
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")

if __name__ == "__main__":
    true_labels = range(100)
    tgt_split = "repair"
    exp_fl_save_path = "./exp-c100c-fl-4_proba_diff.csv"
    df = pd.read_csv(exp_fl_save_path)
    print(f"df.shape: {df.shape}")
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
    # grouped_df = df_extracted.groupby(["label", "op", "fl_method", "fl_target"])["diff_proba"].mean().reset_index()
    grouped_df = (
        df_extracted
        .groupby(["label", "op", "fl_method", "fl_target", "num_weight"])
        .agg(mean_diff_proba=("diff_proba", "mean"), std_diff_proba=("diff_proba", "std"))
        .reset_index()
    )
    # grouped_df = df_extracted.groupby(["label", "op", "fl_method", "fl_target"])["diff_proba"].mean().reset_index()
    grouped_df["mean_diff_proba"] = grouped_df["mean_diff_proba"] * 100
    print(f"grouped_df.shape: {grouped_df.shape}")
    
    for num_weight_value in grouped_df["num_weight"].unique():
        grouped_subset = grouped_df[grouped_df["num_weight"] == num_weight_value]
        plt.figure(figsize=(8, 6))
        filename = f"./exp-c100c-fl-5_proba_diff_wnum{num_weight_value}"
        # 寒色系 (neuron) と暖色系 (weight)、グレー系 (random) の色リスト
        color_map = {
            "neuron": ["blue", "cyan", "purple", "lightblue"],
            "weight": ["red", "orange", "green", "darkred"],
            "random": ["gray", "darkgray", "lightgray"]
        }
        used_colors = set()
        for target in grouped_subset["fl_target"].unique():
            for i, (fl_method, fl_target) in enumerate(product(grouped_subset["fl_method"].unique(), [target])):
                # 特定のfl_method, fl_targetの行だけ抜き出し (2xクラス数 行)
                subset = grouped_subset[(grouped_subset["fl_method"] == fl_method) & (grouped_subset["fl_target"] == fl_target)]
                # ラベルごとに，enh/supでmean_diff_probaが最大の行を抜き出し (クラス数 行)
                subset = subset.groupby(["label"]).apply(lambda group: group.loc[group["mean_diff_proba"].idxmax()])
                print(subset.shape)
                print(subset["op"].value_counts())
                if len(subset) == 0:
                    continue
                # カラー選択のロジック
                if "random" in fl_method:
                    color_list = color_map["random"]
                elif "neuron" in fl_target:
                    color_list = color_map["neuron"]
                else:
                    color_list = color_map["weight"]
                # 未使用の色を選ぶ
                color = next(color for color in color_list if color not in used_colors)
                used_colors.add(color)
                # subset["mean_diff_proba"]で0以上の数を数える
                n_pos = len(subset[subset["mean_diff_proba"] > 0])
                plt.plot(subset["label"], subset["mean_diff_proba"], label=f"{fl_method} ({fl_target}), n_pos={n_pos}", color=color, alpha=0.66)
                mean_of_means = subset["mean_diff_proba"].mean()
                plt.axhline(mean_of_means, color=color, linestyle="--")
        plt.axhline(0, color="black", linestyle="-")
        plt.xlabel("Class Label")
        plt.ylabel("Average Change of Probability (%)")
        plt.legend(loc='lower right', bbox_to_anchor=(1, 0))
        plt.savefig(filename + ".pdf", dpi=300, bbox_inches="tight")
        plt.savefig(filename + ".png", dpi=300, bbox_inches="tight")
        print(f"Saved to {filename}.png/pdf")