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
    parser.add_argument("--separate_by_target", action="store_true", default=False)
    args = parser.parse_args()
    separate_by_target = args.separate_by_target
    
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
    
    
    if not separate_by_target:
        for target in grouped_df["fl_target"].unique():
            filename = f"./exp-fl-4_{ds}_proba_diff_{target}"
            # op, fl_method, fl_targetの組み合わせごとにdiff_probaの平均をプロット
            plt.figure(figsize=(8, 6))
            color_list = ["red", "blue", "orange", "lightblue", "yellow", "green", "purple", "brown", "pink", "gray", "cyan", "magenta"]
            for i, (fl_method, fl_target) in enumerate(product(grouped_df["fl_method"].unique(), [target])):
                # 特定のfl_method, fl_targetの行だけ抜き出し (2xクラス数 行)
                subset = grouped_df[(grouped_df["fl_method"] == fl_method) & (grouped_df["fl_target"] == fl_target)]
                # ラベルごとに，enh/supでmean_diff_probaが最大の行を抜き出し (クラス数 行)
                subset = subset.groupby(["label"]).apply(lambda group: group.loc[group["mean_diff_proba"].idxmax()])
                print(subset.shape)
                print(subset["op"].value_counts())
                if len(subset) == 0:
                    continue
                # subset["mean_diff_proba"]で0以上の数を数える
                n_pos = len(subset[subset["mean_diff_proba"] > 0])
                plt.plot(subset["label"], subset["mean_diff_proba"], label=f"{fl_method} ({fl_target}), n_pos={n_pos}", color=color_list[i], alpha=0.66)
                mean_of_means = subset["mean_diff_proba"].mean()
                plt.axhline(mean_of_means, color=color_list[i], linestyle="--")
            plt.axhline(0, color="black", linestyle="-")
            plt.xlabel("Class Label")
            plt.ylabel("Average Change of Probability (%)")
            plt.legend(loc='lower right', bbox_to_anchor=(1, 0))
            plt.savefig(filename + ".pdf", dpi=300, bbox_inches="tight")
            plt.savefig(filename + ".png", dpi=300, bbox_inches="tight")
            # plt.show()
    else:
        plt.figure(figsize=(8, 6))
        filename = f"./exp-fl-4_{ds}_proba_diff"
        # 寒色系 (neuron) と暖色系 (weight)、グレー系 (random) の色リスト
        color_map = {
            "neuron": ["blue", "cyan", "purple", "lightblue"],
            "weight": ["red", "orange", "green", "darkred"],
            "random": ["gray", "darkgray", "lightgray"]
        }
        used_colors = set()
        for target in grouped_df["fl_target"].unique():
            for i, (fl_method, fl_target) in enumerate(product(grouped_df["fl_method"].unique(), [target])):
                # 特定のfl_method, fl_targetの行だけ抜き出し (2xクラス数 行)
                subset = grouped_df[(grouped_df["fl_method"] == fl_method) & (grouped_df["fl_target"] == fl_target)]
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