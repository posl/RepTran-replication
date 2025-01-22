import os, sys
sys.path.append("../src")
import argparse
import pandas as pd
import torch
import pickle
from datasets import load_from_disk
from utils.constant import ViTExperiment
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")

# 予測ラベルを返す関数
def pred_to_label(pred):
    proba = torch.nn.functional.softmax(torch.tensor(pred.predictions), dim=-1)
    return proba.cpu().numpy().argmax(axis=1)

if __name__ == "__main__":
    # ds_nameをargparseで受けとる
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str)
    args = parser.parse_args()
    ds_name = args.ds

    # laod datasets
    pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR
    pred_out_dir = os.path.join(pretrained_dir, "pred_results", "PredictionOutput")
    dataset_dir = ViTExperiment.DATASET_DIR
    ds = load_from_disk(os.path.join(dataset_dir, ds_name))
    dsc = load_from_disk(os.path.join(dataset_dir, ds_name+'c'))

    f1_dict = {}
    # loop for ds and dsc
    for d in [ds, dsc]:
        # loop for columns of the dataset
        for key in d.keys():
            print(f"key: {key}")
            # 予測結果の読み込み
            filename = f"{key}_pred.pkl" if d == ds else f"{ds_name}c_{key}_pred.pkl"
            with open(os.path.join(pred_out_dir, filename), "rb") as f:
                pred = pickle.load(f)
            true_labels = pred.label_ids
            pred_labels = pred_to_label(pred)
            f1_dict[key] = f1_score(true_labels, pred_labels, average=None)
    # make df contain f1 scores
    f1_df = pd.DataFrame(f1_dict).T
    
    # draw heatmap for f1_df
    plt.figure(figsize=(15, 5))
    sns.heatmap(f1_df, annot=False, cmap="BuGn", cbar=True)
    plt.xlabel("Class")
    plt.ylabel("Dataset")
    plt.title("F1 score per class")
    plt.savefig(os.path.join(pretrained_dir, "pred_results", "heatmap_f1_score.pdf"), dpi=300)