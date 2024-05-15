"""
ノイズデータでfine-tuneした前後での, repaired sample, broken sampleを集計する
"""

import os, sys, math
import numpy as np
import argparse
import pickle
from utils.constant import ViTExperiment
from utils.helper import get_corruption_types
from utils.vit_util import count_pred_change
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("dark")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str)
    parser.add_argument("--ct", type=str, help="corruption type for fine-tuned dataset. when set to None, original model is used", default=None)
    parser.add_argument('--severity', type=int, help="severity of corruption (integer from 0 to 4).", default=4)
    args = parser.parse_args()
    ds_name = args.ds
    ori_ct = args.ct
    severity = args.severity
    print(f"ds_name: {ds_name}, ori_ct: {ori_ct}, severity: {severity}")
    
    ct_list = get_corruption_types()
    dataset_dir = ViTExperiment.DATASET_DIR
    pretrained_dir = getattr(ViTExperiment, ds_name.rstrip('c')).OUTPUT_DIR
    # prediction results of original models
    org_pred_dir = os.path.join(pretrained_dir, "pred_results_divided_corr", "PredictionOutput")
    # prediction results of fine-tuned models with noisy dataset
    ft_pred_dir = os.path.join(pretrained_dir, f"{ori_ct}_severity{severity}", "pred_results", "PredictionOutput")

    # 予測結果を見に行くファイル名のリスト
    filenames = [f"ori_test_pred.pkl"] + [f"{ds_name}_{ct}_pred.pkl" for ct in ct_list]
    for filename in filenames:
        print(f"\nloading from {filename}")
        # org modelのresult
        with open(os.path.join(org_pred_dir, filename), "rb") as f:
            org_pred = pickle.load(f)
        # corrup modelのresult
        with open(os.path.join(ft_pred_dir, filename), "rb") as f:
            ft_pred = pickle.load(f)
        # ft前後の予測結果からrepaired, non-repaired, broken, non-brokenのデータのインデックスをまとめる
        res_dict = count_pred_change(org_pred, ft_pred)
        # res_dictの各値のリストの長さを取得
        res_len = {k: len(v) for k, v in res_dict.items()}
        # print(res_len)
        # repaired ratioとbroken ratioを計算
        repaired_ratio = res_len["repaired"] / (res_len["repaired"] + res_len["non_repaired"])
        broken_ratio = res_len["broken"] / (res_len["broken"] + res_len["non_broken"])
        # repaired ratioとbroken ratioを実際のサンプル数と合わせて表示
        print(f"repaired ratio: {repaired_ratio:.2f} ({res_len['repaired']} / {res_len['repaired'] + res_len['non_repaired']})")
        print(f"broken ratio: {broken_ratio:.2f} ({res_len['broken']} / {res_len['broken'] + res_len['non_broken']})")