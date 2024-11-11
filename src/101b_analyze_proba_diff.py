import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
from collections import defaultdict
from utils.constant import ViTExperiment
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")

def main(ds_name, k, tgt_rank, n, misclf_type, fpfn, run_all=True):
    # print(f"Dataset: {ds_name}, k: {k}, tgt_rank: {tgt_rank}, n: {n}, misclf_type: {misclf_type}, fpfn: {fpfn}")
    pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR.format(k=k)
    # FL情報の保存場所
    save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_weights_location")
    if misclf_type == "all":
        save_dir = os.path.join(pretrained_dir, f"all_weights_location")
    if fpfn is not None and misclf_type == "tgt":
        save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_{fpfn}_weights_location")

    results = []

    # plt.figure(figsize=(8, 6))
    diff_proba = defaultdict(lambda: np.array([]))
    diff_proba_mean = defaultdict(float)
    for op in ["enh", "sup"]:
        for fl_method in ["vdiff", "random"]:
            proba_save_dir = os.path.join(save_dir, f"proba_n{n}_{fl_method}")

            for tl in true_labels:
                # オリジナルのモデルの予測確率を取得
                ori_proba_save_path = os.path.join(pretrained_dir, "pred_results", f"{tgt_split}_proba_{tl}.npy")
                ori_proba = np.load(ori_proba_save_path)[:, tl] # tlへの予測確率 # 形状は (tlのラベルのサンプル数, ) の1次元配列
                
                # 特定位置だけopで変更したモデルの予測確率を取得
                proba_save_path = os.path.join(proba_save_dir, f"{tgt_split}_proba_{op}_{tl}.npy")
                proba = np.load(proba_save_path)[:, tl] # tlへの予測確率
                # print(f"Op: {op}, True label: {tl}, {tgt_split} proba shape: {proba.shape}, ori proba shape: {ori_proba.shape}")
                assert proba.shape == ori_proba.shape, f"{proba.shape} != {ori_proba.shape}"
                mean_diff = np.mean(proba - ori_proba) # ラベルtlの全サンプルに対する予測確率の変化の平均
                diff_proba[(op, fl_method)] = np.append(diff_proba[(op, fl_method)], mean_diff)
            # NOTE: 絶対値でいいのか？変化するということを言いたいなら絶対値でいいが...
            diff_proba_mean[(op, fl_method)] = np.mean(np.abs(diff_proba[(op, fl_method)]))
            
            results.append({
                "n": n,
                "k": k,
                "tgt_rank": tgt_rank,
                "misclf_type": misclf_type,
                "fpfn": fpfn,
                "fl_method": fl_method,
                "op": op,
                "diff_proba_mean": diff_proba_mean[(op, fl_method)]
            })

            # print(f"op: {op}, fl_method: {fl_method}, diff_proba_mean: {diff_proba_mean[(op, fl_method)]}")
            # 一旦描画してみる
            # if fl_method == "vdiff":
            # plt.plot(diff_proba[fl_method]*100, label=f"{op} ({fl_method})", 
            #         marker="x" if fl_method == "random" else "o", markersize=4,
            #         linestyle="--" if fl_method == "random" else "-")

    return pd.DataFrame(results)

    # plt.axhline(0, color="#BBBBBB", linestyle="--")
    # plt.xlabel("Class Label")
    # plt.ylabel("Average Change of Probability (%)")
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    ds = "c100"
    run_all = True
    true_labels = range(100) if ds == "c100" else None
    tgt_split = "repair"
    k_list = range(5)
    tgt_rank_list = range(1, 6)
    n_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 77, 109]
    misclf_type_list = ["all", "src_tgt", "tgt"]
    fpfn_list = [None, "fp", "fn"]

    # 全ての結果を格納するDataFrame
    all_results = pd.DataFrame()

    for k, tgt_rank, n, misclf_type, fpfn in tqdm(product(k_list, tgt_rank_list, n_list, misclf_type_list, fpfn_list), total=len(k_list)*len(tgt_rank_list)*len(n_list)*len(misclf_type_list)*len(fpfn_list)):
        if (misclf_type == "src_tgt" or misclf_type == "all") and fpfn is not None:
            continue
        # 各パラメータの組み合わせでmain関数から1行分のDataFrameを取得し、all_resultsに連結
        result_df = main(ds, k, tgt_rank, n, misclf_type, fpfn, run_all=run_all)
        all_results = pd.concat([all_results, result_df], ignore_index=True)
    # all_resultsを保存
    all_results.to_csv(f"./{ds}_proba_diff_abs_mean.csv", index=False)