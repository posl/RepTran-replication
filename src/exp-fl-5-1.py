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

    # pretrained modelのディレクトリ
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
    diff_wbef = torch.abs(wbef_list[1] - wbef_list[0])
    diff_waft = torch.abs(waft_list[1] - waft_list[0])
    diff_wbef = diff_wbef.cpu().detach().numpy()
    diff_waft = diff_waft.cpu().detach().numpy()
    
    # wbef, waftそれぞれで変化の絶対値が大きい順にインデックスを取得
    pos_before = np.unravel_index(np.argsort(-diff_wbef, axis=None), diff_wbef.shape)  # 大きい順
    pos_after = np.unravel_index(np.argsort(-diff_waft, axis=None), diff_waft.shape)  # 大きい順

    # インデックスをタプルのリストに変換
    pos_before_list = np.array(list(zip(pos_before[0], pos_before[1])))
    pos_after_list = np.array(list(zip(pos_after[0], pos_after[1])))
    print(f"pos_before_list: {pos_before_list.shape}")
    print(f"pos_after_list: {pos_after_list.shape}")

    # インデックスリストをnpyファイルとして保存
    np.save(os.path.join(exp_dir, f"fl_gt_before_{ds_name}_fold{k}.npy"), pos_before_list)
    np.save(os.path.join(exp_dir, f"fl_gt_after_{ds_name}_fold{k}.npy"), pos_after_list)
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