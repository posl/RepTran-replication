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
    ds = "c100"
    true_labels = range(100) if ds == "c100" else None
    tgt_split = "repair"
    save_path = f"./exp-fl-1_{ds}_proba_diff.csv"
    df = pd.read_csv(save_path)
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
    
    for target in grouped_df["fl_target"].unique():
        # op, fl_method, fl_targetの組み合わせごとにdiff_probaの平均をプロット
        plt.figure(figsize=(8, 6))
        color_list = ["red", "blue", "orange", "lightblue", "green", "yellow", "purple", "brown", "pink", "gray"]
        for i, (fl_method, fl_target, op) in enumerate(
            product(grouped_df["fl_method"].unique(), [target], grouped_df["op"].unique())
        ):
            subset = grouped_df[(grouped_df["op"] == op) & (grouped_df["fl_method"] == fl_method) & (grouped_df["fl_target"] == fl_target)]
            plt.plot(subset["label"], subset["mean_diff_proba"], label=f"{fl_method}, {op}, {fl_target}", color=color_list[i])
            mean_of_means = subset["mean_diff_proba"].mean()
            plt.axhline(mean_of_means, color=color_list[i], linestyle="--")
        plt.axhline(0, color="black", linestyle="-")
        plt.xlabel("Class Label")
        plt.ylabel("Average Change of Probability (%)")
        plt.legend()
        plt.savefig(f"./exp-fl-1_{ds}_proba_diff_{target}.pdf", dpi=300, bbox_inches="tight")
        plt.savefig(f"./exp-fl-1_{ds}_proba_diff_{target}.png", dpi=300, bbox_inches="tight")
        # plt.show()