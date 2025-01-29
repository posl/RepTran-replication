import os, sys, time, pickle, json, math
import shutil
from tqdm import tqdm
from collections import defaultdict
from itertools import product
import numpy as np
import pandas as pd
from utils.constant import ViTExperiment, Experiment1, ExperimentRepair1
from utils.log import set_exp_logging
from logging import getLogger
from datasets import load_from_disk

logger = getLogger("base_logger")
tgt_pos = ViTExperiment.CLS_IDX

def get_save_dir(pretrained_dir, tgt_rank, misclf_type, fpfn):
    save_dir = os.path.join(
        pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_weights_location"
    )
    if misclf_type == "all":
        save_dir = os.path.join(pretrained_dir, f"all_weights_location")
    if fpfn is not None and misclf_type == "tgt":
        save_dir = os.path.join(
            pretrained_dir,
            f"misclf_top{tgt_rank}",
            f"{misclf_type}_{fpfn}_weights_location",
        )
    return save_dir

def main(ds_name, k, tgt_rank, misclf_type, fpfn, fl_method, n):
    print(f"ds_name: {ds_name}, fold_id: {k}, tgt_rank: {tgt_rank}, misclf_type: {misclf_type}, fpfn: {fpfn}, fl_method: {fl_method}")
    
    # 定数
    tgt_split = "repair" # NOTE: we only use repair split for repairing
    tgt_layer = 11 # NOTE: we only use the last layer for repairing
    
    # datasetをロード (true_labelsが欲しいので)
    ds_dirname = f"{ds_name}_fold{k}"
    ds = load_from_disk(os.path.join(ViTExperiment.DATASET_DIR, ds_dirname))
    label_col = "fine_label"
    true_labels = ds[tgt_split][label_col]
    label_list = range(100)
    
    # ニューロンへの介入の方法のリスト
    op_list = ["enhance", "suppress"]
    
    # pretrained modelのディレクトリ
    pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR.format(k=k)
    # 結果とかログの保存先
    save_dir = get_save_dir(pretrained_dir, tgt_rank, misclf_type, fpfn)
    # 予測確率のリストの保存先
    proba_save_dir = os.path.join(save_dir, f"exp-fl-2_proba_n{n}_{fl_method}")
        
    # このpythonのファイル名を取得
    this_file_name = os.path.basename(__file__).split(".")[0]
    exp_name = f"exp-fl-2_{this_file_name}"
    # loggerの設定をして設定情報を表示
    logger = set_exp_logging(exp_dir=save_dir, exp_name=exp_name)
    logger.info(f"ds_name: {ds_name}, fold_id: {k}, tgt_rank: {tgt_rank}, misclf_type: {misclf_type}, tgt_split: {tgt_split}, tgt_layer: {tgt_layer}")
    
    results = []
    
    for op in op_list:
        print(f"op: {op}")
        for tl in label_list:
            # オリジナルのモデルの予測確率を取得
            ori_proba_save_path = os.path.join(pretrained_dir, "pred_results", f"{tgt_split}_proba_{tl}.npy")
            ori_proba = np.load(ori_proba_save_path)[:, tl]  # tlへの予測確率 # 形状は (tlのラベルのサンプル数, ) の1次元配列
            # 特定位置だけopで変更したモデルの予測確率を取得
            proba_save_path = os.path.join(proba_save_dir, f"{tgt_split}_proba_{op}_{tl}.npy")
            proba = np.load(proba_save_path)[:, tl]  # tlへの予測確率
            assert (proba.shape == ori_proba.shape), f"{proba.shape} != {ori_proba.shape}"
            mean_diff = np.mean(proba - ori_proba)  # ラベルtlの全サンプルに対する予測確率の変化の平均
            
            results.append(
                {
                    "n": n,
                    "num_weight": 768 * n if fl_method == "ig" else None, #TODO 
                    "k": k,
                    "tgt_rank": tgt_rank,
                    "misclf_type": misclf_type,
                    "fpfn": fpfn,
                    "fl_target": "neuron" if fl_method == "ig" else "weight",
                    "fl_method": fl_method,
                    "op": op,
                    "label": tl,
                    "diff_proba": mean_diff,
                }
            )
    return pd.DataFrame(results)

if __name__ == "__main__":
    ds = "c100"
    k_list = range(5)
    tgt_rank_list = range(1, 6)
    misclf_type_list = ["all", "src_tgt", "tgt"]
    fpfn_list = [None, "fp", "fn"]
    fl_method_list = ["ig", "bl"]
    # n_list = [Experiment1.NUM_IDENTIFIED_NEURONS, ExperimentRepair1.NUM_IDENTIFIED_NEURONS]
    n_list = [ExperimentRepair1.NUM_IDENTIFIED_NEURONS]
    
    # 全ての結果を格納するDataFrame
    all_results = pd.DataFrame()
    
    for k, tgt_rank, misclf_type, fpfn, fl_method, n in product(k_list, tgt_rank_list, misclf_type_list, fpfn_list, fl_method_list, n_list):
        if (misclf_type == "src_tgt" or misclf_type == "all") and fpfn is not None: # misclf_type == "src_tgt" or "all"の時はfpfnはNoneだけでいい
            continue
        if misclf_type == "all" and tgt_rank != 1: # misclf_type == "all"の時にtgt_rankは関係ないのでこのループもスキップすべき
            continue
        if misclf_type != "all" and fl_method == "ig": # igは誤分類のタイプごとには計算できない
            continue
        result_df = main(ds, k, tgt_rank, misclf_type, fpfn, fl_method, n)
        all_results = pd.concat([all_results, result_df], ignore_index=True)
        print(f"all_results.shape: {all_results.shape}")
    # all_resultsを保存
    save_path = f"./exp-fl-2_{ds}_proba_diff.csv"
    all_results.to_csv(save_path, index=False)
