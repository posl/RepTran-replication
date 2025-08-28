import os, sys, time, pickle, json, math
import shutil
import torch
from tqdm import tqdm
from collections import defaultdict
from itertools import product
import numpy as np
import pandas as pd
from utils.helper import get_device, json2dict
from utils.vit_util import ViTFromLastLayer
from utils.de import set_new_weights, check_new_weights
from utils.constant import ViTExperiment, Experiment3
from utils.log import set_exp_logging
from logging import getLogger
from datasets import load_from_disk
from transformers import ViTForImageClassification
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")

def get_intermediate_weights(model, tgt_layer):
    """
    model: ViTForImageClassification
    tgt_layer: int
    """
    wbef = model.vit.encoder.layer[tgt_layer].intermediate.dense.weight
    waft = model.vit.encoder.layer[tgt_layer].output.dense.weight
    return wbef, waft

logger = getLogger("base_logger")
tgt_pos = ViTExperiment.CLS_IDX
exp_dir = "./exp-fl-5"
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

device = get_device()
ds_name = "c100"
num_fold = 5
tgt_layer = 11 # ViT-baseの最終層

for k in range(num_fold):
    print(f"ds_name: {ds_name}, fold_id: {k}")

    # datasetをロード (true_labelsが欲しいので)
    ds_dirname = f"{ds_name}_fold{k}"
    ds = load_from_disk(os.path.join(ViTExperiment.DATASET_DIR, ds_dirname))
    label_col = "fine_label"

    # Directory for pretrained model
    pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR.format(k=k)
    
    # 1エポック目のモデルと2エポック目のモデルをそれぞれロード
    model1 = ViTForImageClassification.from_pretrained(os.path.join(pretrained_dir, "checkpoint-1250"))
    model2 = ViTForImageClassification.from_pretrained(os.path.join(pretrained_dir, "checkpoint-2500"))

    # 各モデルにおける対象範囲に含まれる重み行列を取り出す
    wbef_list, waft_list = [], []
    for model in [model1, model2]:
        model.to(device)
        model.eval()
        wbef, waft = get_intermediate_weights(model, tgt_layer)
        wbef_list.append(wbef)
        waft_list.append(waft)
    # 重み行列の差分の絶対値 (変化量だけに興味があり符号には興味がない) を計算
    diff_wbef = torch.abs(wbef_list[1] - wbef_list[0]).cpu().detach().numpy()
    diff_waft = torch.abs(waft_list[1] - waft_list[0]).cpu().detach().numpy()
    shape_bef = diff_wbef.shape
    shape_aft = diff_waft.shape
    split_idx = diff_wbef.size
    # 統合してランキングを作成
    combined_diff = np.concatenate([diff_wbef.flatten(), diff_waft.flatten()])
    sorted_indices = np.argsort(combined_diff)[::-1]  # 降順でランキング作成, i番目はcombined_diffがi番目に大きいデータのインデックス
    ranks = np.zeros_like(sorted_indices) # 1次元
    ranks[sorted_indices] = np.arange(1, len(combined_diff) + 1)  # 降順, i番目はcombined_diff[i]の順位
    # 確認
    # print(combined_diff[sorted_indices[:13]])
    # print(ranks[sorted_indices[:13]])
    # print(sorted_indices[:13])
    # for si in sorted_indices[:13]:
    #     if si < split_idx:
    #         print(f"bef: {np.unravel_index(si, shape_bef)}")
    #     else:
    #         print(f"aft: {np.unravel_index(si - split_idx, shape_aft)}")
    # befとaftに分類
    bef_indices, bef_ranks = [], []
    aft_indices, aft_ranks = [], []
    
    
    for idx, rank in enumerate(ranks): # NOTE: ここでのrankはcombined_diff[idx]のデータの順位
        if idx < split_idx:
            bef_indices.append(np.unravel_index(idx, shape_bef))
            bef_ranks.append(rank)
        else:
            adjusted_idx = idx - split_idx
            aft_indices.append(np.unravel_index(adjusted_idx, shape_aft))
            aft_ranks.append(rank)
    # nparrayに変更
    bef_indices = np.array(bef_indices)
    aft_indices = np.array(aft_indices)
    bef_ranks = np.array(bef_ranks)
    aft_ranks = np.array(aft_ranks)
    print(f"len(bef_indices): {len(bef_indices)}, len(aft_indices): {len(aft_indices)}")
    print(f"len(bef_ranks): {len(bef_ranks)}, len(aft_ranks): {len(aft_ranks)}")
    
    # スコアのランキングの昇順（1位が一番大きいのでスコアの降順）にソート
    sorted_bef = np.argsort(bef_ranks)  # ランク昇順
    bef_indices = bef_indices[sorted_bef]
    bef_ranks = bef_ranks[sorted_bef]

    sorted_aft = np.argsort(aft_ranks)  # ランク昇順
    aft_indices = aft_indices[sorted_aft]
    aft_ranks = aft_ranks[sorted_aft]
    
    # 全体のランクが30位以内の全てを表示
    print("Top 30: (just for checking)")
    rank_mask_bef = bef_ranks <= 30
    rank_mask_aft = aft_ranks <= 30
    print(f"bef_indices: {bef_indices[rank_mask_bef]}")
    print(f"aft_indices: {aft_indices[rank_mask_aft]}")
    print(f"bef_ranks: {bef_ranks[rank_mask_bef]}")
    print(f"aft_ranks: {aft_ranks[rank_mask_aft]}")
    
    # ランキングをSave
    np.save(os.path.join(exp_dir, f"fl_gt_before_{ds_name}_fold{k}_rank.npy"), bef_ranks)
    np.save(os.path.join(exp_dir, f"fl_gt_after_{ds_name}_fold{k}_rank.npy"), aft_ranks)
    print(f"Saved: fl_gt_before_{ds_name}_fold{k}_rank.npy")
    print(f"Saved: fl_gt_after_{ds_name}_fold{k}_rank.npy")
    
    # インデックスリストをnpyファイルとしてSave
    np.save(os.path.join(exp_dir, f"fl_gt_before_{ds_name}_fold{k}.npy"), bef_indices)
    np.save(os.path.join(exp_dir, f"fl_gt_after_{ds_name}_fold{k}.npy"), aft_indices)
    print(f"Saved: fl_gt_before_{ds_name}_fold{k}.npy")
    print(f"Saved: fl_gt_after_{ds_name}_fold{k}.npy")
    
    # 上位3%の閾値を計算
    threshold_bef = np.percentile(np.abs(diff_wbef), 97)  # diff_wbef の上位3%
    threshold_aft = np.percentile(np.abs(diff_waft), 97)  # diff_waft の上位3%
    
    # データを整形
    data = {
        "Value": np.concatenate([diff_wbef.flatten(), diff_waft.flatten()]),
        "Type": ["diff_wbef"] * diff_wbef.size + ["diff_waft"] * diff_waft.size
    }
    df = pd.DataFrame(data)
    # ヒストグラムの描画
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="Value", hue="Type", bins=100, kde=True, alpha=0.5)
    
    # diff_wbef の上位3%閾値に点線を引く
    plt.axvline(threshold_bef, color="blue", linestyle="--", label=f"Top 3% diff_wbef: {threshold_bef:.3f}")
    # diff_waft の上位3%閾値に点線を引く
    plt.axvline(threshold_aft, color="orange", linestyle="--", label=f"Top 3% diff_waft: {threshold_aft:.3f}")

    # グラフの設定
    plt.xlabel("Difference Magnitude")
    plt.ylabel("Frequency")
    plt.title("Histogram of Weight Differences")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # グラフの表示
    plt.savefig(os.path.join(exp_dir, f"hist_diff_weights_{ds_name}_fold{k}.png"), dpi=300, bbox_inches="tight")
    print(f"Saved: {os.path.join(exp_dir, f'hist_diff_weights_{ds_name}_fold{k}.png')}")