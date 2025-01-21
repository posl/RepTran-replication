import os, sys, time, pickle, json, math
from tqdm import tqdm
from collections import defaultdict
from itertools import product
import numpy as np
import pandas as pd
from utils.helper import get_device, json2dict
from utils.vit_util import identfy_tgt_misclf
from utils.constant import ViTExperiment
import torch
import torch.optim as optim

# デバイス (cuda, or cpu) の取得
device = get_device()

def generate_random_array(a):
    """
    配列 a と同じ形状のランダム配列 b を生成。
    - 1軸目: [0, a.shape[0]] の整数
    - 2軸目: [0, a.shape[1]] の整数
    """
    b = np.zeros_like(a, dtype=int)
    b[:, 0] = np.random.randint(0, a.shape[0] + 1, size=a.shape[0])
    b[:, 1] = np.random.randint(0, a.shape[1] + 1, size=a.shape[0])
    return b

def main(ds_name, k, tgt_rank_list, misclf_type, fpfn, n):
    ts = time.perf_counter()
    
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
    
    # FL_GTのロード
    fl_gt_before_path = os.path.join("./exp-fl-5", f"fl_gt_before_{ds}_fold{k}.npy")
    fl_gt_after_path = os.path.join("./exp-fl-5", f"fl_gt_after_{ds}_fold{k}.npy")
    fl_gt_before = np.load(fl_gt_before_path)
    fl_gt_after = np.load(fl_gt_after_path)
    print(f"fl_gt_before.shape: {fl_gt_before.shape}, fl_gt_after.shape: {fl_gt_after.shape}")
    
    # tgt_rankの誤分類情報を取り出す
    misclf_info_dir = os.path.join(exp_dir, "misclf_info")
    misclf_pair, tgt_label, tgt_mis_indices = identfy_tgt_misclf(misclf_info_dir, tgt_split=tgt_split, tgt_rank=tgt_rank, misclf_type=misclf_type, fpfn=fpfn)
    indices_to_incorrect = tgt_mis_indices
    if misclf_type == "src_tgt":
        slabel, tlabel = misclf_pair
    elif misclf_type == "tgt":
        tlabel = tgt_label
        
    w_num = n = "All" if n is None else n
    
    fl_method_list = ["ours", "bl", "random"]
    for fl_method in fl_method_list:
        # location informationの保存先
        if fl_method == "random":
            # fl_gtと同じ形状の乱数を生成
            pos_before = generate_random_array(fl_gt_before)
            pos_after = generate_random_array(fl_gt_after)
        else:
            assert fl_method in ["ours", "bl"], f"fl_method={fl_method} is not supported."
            if fl_method == "ours":
                continue
                location_filename = f"exp-fl-5_location_n{n}_w{w_num}_weight.npy"
                rank_filename = f"exp-fl-5_location_n{n}_w{w_num}_weight_rank.npy"
            elif fl_method == "bl":
                location_filename = f"exp-fl-5_location_n{n}_weight_bl.npy"
                rank_filename = f"exp-fl-5_location_n{n}_weight_bl_rank.npy"
            # 重み位置情報のロード
            location_path = os.path.join(location_save_dir, location_filename)
            rank_path = os.path.join(location_save_dir, rank_filename)
            pos_before, pos_after = np.load(location_path)
            rank_before, rank_after = np.load(rank_path)
        # 位置情報のshapeチェック
        print(f"pos_before.shape: {pos_before.shape}, pos_after.shape: {pos_after.shape}")
        assert pos_before.shape == fl_gt_before.shape, f"{pos_before.shape} != {fl_gt_before.shape}"
        assert pos_after.shape == fl_gt_after.shape, f"{pos_after.shape} != {fl_gt_after.shape}"
        print(f"rank_before: {rank_before}, rank_after: {rank_after}")
        exit()

if __name__ == "__main__":
    ds = "c100"
    k_list = range(5)
    tgt_rank_list = range(1, 6)
    misclf_type_list = ["all", "src_tgt", "tgt"]
    fpfn_list = [None, "fp", "fn"]
    n_list = [None]
    for k, tgt_rank, misclf_type, fpfn, n in product(k_list, tgt_rank_list, misclf_type_list, fpfn_list, n_list):
        print(f"Start: ds={ds}, k={k}, tgt_rank={tgt_rank}, misclf_type={misclf_type}, fpfn={fpfn}")
        if (misclf_type == "src_tgt" or misclf_type == "all") and fpfn is not None: # misclf_type == "src_tgt" or "all"の時はfpfnはNoneだけでいい
            continue
        if misclf_type == "all" and tgt_rank != 1: # misclf_type == "all"の時にtgt_rankは関係ないのでこのループもスキップすべき
            continue
        main(ds, k, tgt_rank, misclf_type, fpfn, n=n)