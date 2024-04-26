import os, sys, math
import numpy as np
import argparse
import pickle
from datasets import load_from_disk
from utils.constant import ViTExperiment
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("dark")

def plot_accuracy(ori_acc_dict, acc_dict):
    save_parth = os.path.join(ViTExperiment.OUTPUT_DIR, "pred_results", "accuracy_per_dataset.pdf")
    datasets = list(acc_dict.keys())
    accuracies = list(acc_dict.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(datasets, accuracies, color='skyblue')
    # plot hline for original accuracy
    for k, ori_acc in ori_acc_dict.items():
        color = "black" if k == "train" else "red"
        plt.axhline(y=ori_acc, color=color, linestyle='--', label=k, linewidth=1)
    plt.legend()
    # legendの位置をプロットの外にする
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Dataset')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_parth, dpi=300)

if __name__ == "__main__":
    pretrained_dir = ViTExperiment.OUTPUT_DIR
    dataset_dir = ViTExperiment.DATASET_DIR
    pred_out_dir = os.path.join(pretrained_dir, "pred_results", "PredictionOutput")
    c10 = load_from_disk(os.path.join(dataset_dir, "c10"))
    c10c = load_from_disk(os.path.join(dataset_dir, "c10c"))
    # dict to store the accuracy for each dataset
    c10_acc_dict = {}
    c10c_acc_dict = {}
    # loop for c10 and c10c
    for ds in [c10, c10c]:
        # loop for columns of the dataset
        for key in ds.keys():
            print(f"key: {key}")
            # 予測結果の読み込み
            filename = f"{key}_pred.pkl" if ds == c10 else f"c10c_{key}_pred.pkl"
            with open(os.path.join(pred_out_dir, filename), "rb") as f:
                pred = pickle.load(f)
            # 予測結果のメトリクス表示
            print(f'acc: {pred.metrics["test_accuracy"]}, f1: {pred.metrics["test_f1"]}')
            if ds == c10:
                c10_acc_dict[key] = pred.metrics["test_accuracy"]["accuracy"]
            elif ds == c10c:
                c10c_acc_dict[key] = pred.metrics["test_accuracy"]["accuracy"]
    plot_accuracy(c10_acc_dict, c10c_acc_dict)