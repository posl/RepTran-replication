import os, sys, time, pickle, json, math
from typing import Tuple
from tqdm import tqdm
from collections import defaultdict
from itertools import product
import numpy as np
import pandas as pd
from utils.helper import get_device, json2dict
from utils.vit_util import identfy_tgt_misclf
from utils.constant import ViTExperiment, Experiment1
import matplotlib.pyplot as plt
import seaborn as sns

NUM_IDENTIFIED_NEURONS = Experiment1.NUM_IDENTIFIED_NEURONS

# デバイス (cuda, or cpu) の取得
device = get_device()

WBEF_SHAPE = (3072, 768)
WAFT_SHAPE = (768, 3072)

def generate_random_indices(shape):
    """
    指定された形状のランダム配列を生成します。

    この関数は、指定された形状 (shape) を元にランダムな整数値を生成し、
    形状に基づいて展開された 2 次元の配列を返します。
    各行は 2 つの列からなり、それぞれ次元ごとのランダムなインデックスを表します。

    Args:
        shape (tuple): ランダム配列を生成する元の形状 (行数, 列数)。

    Returns:
        numpy.ndarray: ランダムに生成された配列 (shape[0] * shape[1], 2)。
                       各行は元の形状のインデックスに対応する (row, column) のペア。
    """
    total_elements = shape[0] * shape[1]
    row_indices = np.repeat(np.arange(shape[0]), shape[1])
    col_indices = np.tile(np.arange(shape[1]), shape[0])
    all_indices = np.column_stack((row_indices, col_indices))
    # シャッフルしてランダムな順序に
    np.random.shuffle(all_indices)
    # ランダムなユニークインデックスを生成
    b = all_indices[:total_elements]
    return b

# def generate_random_rank(n):
#     if n % 2 != 0:
#         raise ValueError("n must be even for a,b to have the same length")
#     total_ranks = np.arange(1, n+1)
#     np.random.shuffle(total_ranks)
#     half = n//2
#     a_raw = total_ranks[:half]
#     b_raw = total_ranks[half:]
#     # 昇順ソート
#     a = np.sort(a_raw)
#     b = np.sort(b_raw)
#     return a, b

def generate_random_rank(N):
    """
    o1's suggestion
    """
    perm = np.random.permutation(N)
    rank_array = np.zeros(N, dtype=int)
    for idx, item in enumerate(perm):
        rank_array[item] = idx+1
    return rank_array

def compute_topk_metrics(fl_rank: np.ndarray, gt_rank: np.ndarray, 
                         K_values: list, top_gt: int) -> Tuple[list, list, list]:
    """
    Compute Precision@K, Recall@K, F1@K for each K in K_values.
    
    :param fl_rank:  1D array of length N, where fl_rank[j] is the rank of item j by the FL method (1 = top).
    :param gt_rank:  1D array of length N, where gt_rank[j] is the rank of item j by the ground truth (1 = top).
    :param K_values: list of integer K's for which we compute the metrics.
    :param top_gt:   how many top items in the GT ranking are considered "actual positives".
    :return: (precision_list, recall_list, f1_list), each is a list of length len(K_values).
    """
    # Indices in GT that are "positives" => items with gt_rank <= top_gt
    gt_positive = set(np.where(gt_rank <= top_gt)[0])

    precision_list, recall_list, f1_list = [], [], []

    for K in K_values:
        # Predicted positives: items with fl_rank <= K
        fl_topK = set(np.where(fl_rank <= K)[0])

        tp = len(fl_topK.intersection(gt_positive))
        pred_size = len(fl_topK)
        actual_size = len(gt_positive)

        if pred_size == 0:
            precision = 0.0
        else:
            precision = tp / pred_size

        if actual_size == 0:
            recall = 0.0
        else:
            recall = tp / actual_size

        if (precision + recall) == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return precision_list, recall_list, f1_list

def compute_topk_overlap(fl_rank: np.ndarray, gt_rank: np.ndarray,
                         K_values: list) -> list:
    """
    Compute "top-K overlap" for each K in K_values.
    - top-K overlap = |{i | fl_rank[i] <= K} ∩ {j | gt_rank[j] <= K}| / K
      i.e., the intersection size of top-K sets from FL rank & GT rank.

    :return: overlap_list, length = len(K_values),
             overlap_list[i] = relative count of intersection at K_values[i].
    """

    overlap_list = []
    for K in K_values:
        fl_topK = set(np.where(fl_rank <= K)[0])
        gt_topK = set(np.where(gt_rank <= K)[0])
        overlap_count = len(fl_topK.intersection(gt_topK))
        overlap_list.append(overlap_count / K)
    return overlap_list

def main(ds_name, k, tgt_rank_list, misclf_type, fpfn, n):
    
    # datasetごとに違う変数のセット
    tgt_split = "repair" # NOTE: we only use repair split for repairing
    tgt_layer = 11 # NOTE: we only use the last layer for repairing
    tgt_pos = ViTExperiment.CLS_IDX
    
    # exp-fl-5の結果保存用ディレクトリ
    exp_dir = os.path.join("./exp-fl-5", f"{ds_name}_fold{k}")
    
    # ニューロン/重みの位置情報が保存されたディレクトリ
    if fpfn is not None and misclf_type == "tgt":
        location_save_dir = os.path.join(exp_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_{fpfn}_weights_location")
    elif misclf_type == "all":
        location_save_dir = os.path.join(exp_dir, f"all_weights_location")
    else:
        location_save_dir = os.path.join(exp_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_weights_location")
    
    # ======== Load Ground Truth for before & after =========
    # set paths
    fl_gt_location_before_path = os.path.join("./exp-fl-5", f"fl_gt_before_{ds}_fold{k}.npy")
    fl_gt_location_after_path = os.path.join("./exp-fl-5", f"fl_gt_after_{ds}_fold{k}.npy")
    fl_gt_rank_before_path = os.path.join("./exp-fl-5", f"fl_gt_before_{ds}_fold{k}_rank.npy")
    fl_gt_rank_after_path = os.path.join("./exp-fl-5", f"fl_gt_after_{ds}_fold{k}_rank.npy")
    # load
    fl_gt_before_location = np.load(fl_gt_location_before_path)
    fl_gt_after_location = np.load(fl_gt_location_after_path)
    fl_gt_before_rank = np.load(fl_gt_rank_before_path)
    fl_gt_after_rank = np.load(fl_gt_rank_after_path)
    print(f"fl_gt_before_location.shape: {fl_gt_before_location.shape}, fl_gt_after_location.shape: {fl_gt_after_location.shape}")
    print(f"fl_gt_before_rank.shape: {fl_gt_before_rank.shape}, fl_gt_after_rank.shape: {fl_gt_after_rank.shape}")
    # NOTE: 全ての重みを使う場合：2359296 = 3072 * 768
    # fl_gt_before_location.shape: (2359296, 2), fl_gt_after_location.shape: (2359296, 2)
    # fl_gt_before_rank.shape: (2359296,), fl_gt_after_rank.shape: (2359296,)
    
    # Combine "before" and "after" into one big GT rank array
    fl_gt_location_combined = np.concatenate([fl_gt_before_location, fl_gt_after_location], axis=0)
    fl_gt_rank_combined     = np.concatenate([fl_gt_before_rank,     fl_gt_after_rank],     axis=0)
    print(f"fl_gt_location_combined.shape: {fl_gt_location_combined.shape}, fl_gt_rank_combined.shape: {fl_gt_rank_combined.shape}")
    # So the total N:
    N = fl_gt_rank_combined.size
    print(f"Combined N = {N}")  # e.g. ~4,718,592
    
    # tgt_rankの誤分類情報を取り出す
    misclf_info_dir = os.path.join(exp_dir, "misclf_info")
    misclf_pair, tgt_label, tgt_mis_indices = identfy_tgt_misclf(misclf_info_dir, tgt_split=tgt_split, tgt_rank=tgt_rank, misclf_type=misclf_type, fpfn=fpfn)
    indices_to_incorrect = tgt_mis_indices
    if misclf_type == "src_tgt":
        slabel, tlabel = misclf_pair
    elif misclf_type == "tgt":
        tlabel = tgt_label
        
    if n is None:
        w_num = n = "All"
    else:
        w_num = 8*n*n
    
    fl_method_list = ["ours", "bl", "random"]
    FL_ranks = {}
    for fl_method in fl_method_list:
        # location informationの保存先
        if fl_method == "random":
            # fl_gtと同じ形状の乱数を生成
            pos_before = generate_random_indices(WBEF_SHAPE)
            pos_after = generate_random_indices(WAFT_SHAPE)
            # rank_before, rank_after = generate_random_rank(len(fl_gt_before_rank)+len(fl_gt_after_rank))
            rank_combined = generate_random_rank(len(fl_gt_before_rank) + len(fl_gt_after_rank))
        else:
            assert fl_method in ["ours", "bl"], f"fl_method={fl_method} is not supported."
            if fl_method == "ours":
                location_filename = f"exp-fl-5_location_n{n}_w{w_num}_weight.npy"
                rank_filename = f"exp-fl-5_location_n{n}_w{w_num}_weight_rank.npy"
            elif fl_method == "bl":
                location_filename = f"exp-fl-5_location_nAll_weight_bl.npy"
                rank_filename = f"exp-fl-5_location_nAll_weight_bl_rank.npy"
            # 重み位置情報のロード
            location_path = os.path.join(location_save_dir, location_filename)
            rank_path = os.path.join(location_save_dir, rank_filename)
            pos_before, pos_after = np.load(location_path)
            rank_before, rank_after = np.load(rank_path)
            # Combine FL ranks 
            rank_combined = np.concatenate([rank_before, rank_after], axis=0)
        # 0, 1列目のmin, maxを表示
        # print(f"pos_before: min={pos_before.min(axis=0)}, max={pos_before.max(axis=0)}")
        # print(f"pos_after: min={pos_after.min(axis=0)}, max={pos_after.max(axis=0)}")
        # print(f"rank_before: min={rank_before.min(axis=0)}, max={rank_before.max(axis=0)}")
        # print(f"rank_after: min={rank_after.min(axis=0)}, max={rank_after.max(axis=0)}")
        # 位置情報のshapeチェック
        # print(f"[{fl_method}] pos_before.shape: {pos_before.shape}, pos_after.shape: {pos_after.shape}")
        # print(f"[{fl_method}] rank_before.shape: {rank_before.shape}, rank_after.shape: {rank_after.shape}")
        # assert pos_before.shape == fl_gt_before_location.shape, f"{pos_before.shape} != {fl_gt_before_location.shape}"
        # assert pos_after.shape == fl_gt_after_location.shape, f"{pos_after.shape} != {fl_gt_after_location.shape}"
        
        FL_ranks[fl_method] = rank_combined
        # print(f"[{fl_method}] rank_combined.shape = {rank_combined.shape}")
        # rank_before, rank_afterにuniqueをして重複がないことを確認
        # print(f"[{fl_method}] rank_before: unique={np.unique(rank_before).size}, total={rank_before.size}")
        # print(f"[{fl_method}] rank_after: unique={np.unique(rank_after).size}, total={rank_after.size}")
        
        
        
    # ============== 1) top_gt = 1% of the combined array ==============
    top_gt = int(0.01 * N)
    print(f"top_gt = 1% of {N} => {top_gt}")
    
    # ============== 2) K as {0.1%, 0.5%, 1%, 2%, 5%, 10%} of N =========
    K_fractions = [0.001, 0.005, 0.01, 0.02, 0.05]
    K_values = [int(math.floor(frac * N)) for frac in K_fractions]
    print(f"K_values = {K_values}")
    
    # ============== 3) Plot them as subplots ===========================
    fig, axes = plt.subplots(1, 4, figsize=(20,4))
    
    ret_dict = defaultdict(defaultdict)
    colors = ["b", "r", "g"]  # 3つの手法用
    for idx, method in enumerate(fl_method_list):
        fl_rank = FL_ranks[method]
        # (A) Precision, Recall, F1
        p_list, r_list, f_list = compute_topk_metrics(
            fl_rank=fl_rank,
            gt_rank=fl_gt_rank_combined,
            K_values=K_values,
            top_gt=top_gt
        )
        # (B) top-K overlap
        overlap_list = compute_topk_overlap(
            fl_rank=fl_rank,
            gt_rank=fl_gt_rank_combined,
            K_values=K_values
        )
        ret_dict[method]["precision"] = p_list
        ret_dict[method]["recall"] = r_list
        ret_dict[method]["f1"] = f_list
        ret_dict[method]["overlap"] = overlap_list
        # 手法ごとのリストをprint
        print(f"[{method}] Precision: {p_list}")
        print(f"[{method}] Recall: {r_list}")
        print(f"[{method}] F1: {f_list}")
        print(f"[{method}] Overlap: {overlap_list}")
        
        # 横軸はパーセント表示: e.g. 0.1, 0.5, 1, 2, 5, 10
        xaxis_percent = [f*100 for f in K_fractions]  # => [0.1, 0.5, 1, 2, 5, 10]
        c = colors[idx]
        
        # Precision subplot
        axes[0].plot(xaxis_percent, p_list, marker='o', color=c, label=method, alpha=0.6)
        # Recall subplot
        axes[1].plot(xaxis_percent, r_list, marker='o', color=c, label=method, alpha=0.6)
        # F1 subplot
        axes[2].plot(xaxis_percent, f_list, marker='o', color=c, label=method, alpha=0.6)
        # Overlap subplot
        axes[3].plot(xaxis_percent, overlap_list, marker='o', color=c, label=method, alpha=0.6)
    
    #=== 各subplotのタイトルやラベル ===
    axes[0].set_title('Precision vs. K%')
    axes[0].set_xlabel('K (% of total)')
    axes[0].set_ylabel('Precision')
    axes[0].grid(True)
    axes[0].legend()
    
    axes[1].set_title('Recall vs. K%')
    axes[1].set_xlabel('K (% of total)')
    axes[1].set_ylabel('Recall')
    axes[1].grid(True)
    axes[1].legend()
    
    axes[2].set_title('F1 vs. K%')
    axes[2].set_xlabel('K (% of total)')
    axes[2].set_ylabel('F1')
    axes[2].grid(True)
    axes[2].legend()
    
    axes[3].set_title('Overlap(K) vs. K%')
    axes[3].set_xlabel('K (% of total)')
    axes[3].set_ylabel('Overlap(K)')
    axes[3].grid(True)
    axes[3].legend()

    plt.tight_layout()
    saved_filename = f"fl_score_vs_gt_{top_gt}_ours_n{n}_w{w_num}" if n is not None else f"fl_score_vs_gt_{top_gt}"
    plt.savefig(os.path.join(location_save_dir, saved_filename + ".png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(location_save_dir, saved_filename + ".pdf"), dpi=300, bbox_inches="tight")
    print(f"Saved: {location_save_dir}/{saved_filename}.png/pdf")
    plt.close(fig)
    
    # ret_dictを保存
    with open(os.path.join(location_save_dir, saved_filename + ".json"), "w") as f:
        json.dump(ret_dict, f, indent=4)
    
    print("Done FL comparison for ds=", ds_name, " k=", k, " in method_list=", fl_method_list)
    return ret_dict

if __name__ == "__main__":
    ds = "c100"
    k_list = range(5)
    tgt_rank_list = range(1, 4)
    misclf_type_list = ["all", "src_tgt", "tgt"]
    fpfn_list = [None, "fp", "fn"]
    n_list = [NUM_IDENTIFIED_NEURONS, None]
    for k, tgt_rank, misclf_type, fpfn, n in product(k_list, tgt_rank_list, misclf_type_list, fpfn_list, n_list):
        print(f"Start: ds={ds}, k={k}, tgt_rank={tgt_rank}, misclf_type={misclf_type}, fpfn={fpfn}")
        if (misclf_type == "src_tgt" or misclf_type == "all") and fpfn is not None: # misclf_type == "src_tgt" or "all"の時はfpfnはNoneだけでいい
            continue
        if misclf_type == "all" and tgt_rank != 1: # misclf_type == "all"の時にtgt_rankは関係ないのでこのループもスキップすべき
            continue
        ret_dict = main(ds, k, tgt_rank, misclf_type, fpfn, n=n)