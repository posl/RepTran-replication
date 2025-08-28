import os, sys, math
import numpy as np
import argparse
import pickle
import evaluate
from datasets import load_from_disk
from utils.constant import ViTExperiment
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("dark")

def plot_accuracy(ori_acc_dict, acc_dict, severity):
    filename = "accuracy_per_dataset.pdf" if severity == -1 else f"accuracy_per_dataset_severity{severity}.pdf"
    save_path = os.path.join(getattr(ViTExperiment, ds_name).OUTPUT_DIR, "pred_results", filename)
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
    plt.savefig(save_path, dpi=300)
    print(f"save plot to {save_path}")

# NOTE: /src/dataset/save_dataset.pyからコピーしたcode clone
def get_sublist(original_list, severity):
    """severityに応じて適切な部分リストを取得する"""
    if severity == -1:
        return original_list # すべての要素を取得する
    elif severity >= 0 and severity <= 4:
        start_index = severity * 10000
        end_index = (severity + 1) * 10000 if severity < 4 else None
        return original_list[start_index:end_index]
    else:
        raise ValueError("severity must be an integer in the range 0 to 4 or -1")

if __name__ == "__main__":
    # Accept dataset via argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str)
    parser.add_argument('--severity', type=int, help="severity of corruption (integer from 0 to 4). when set to -1, treat all as one dataset.", default=-1)
    args = parser.parse_args()
    ds_name = args.ds
    severity = args.severity
    print(f"ds_name: {ds_name}")
    
    pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR
    dataset_dir = ViTExperiment.DATASET_DIR
    pred_out_dir = os.path.join(pretrained_dir, "pred_results", "PredictionOutput")
    ds = load_from_disk(os.path.join(dataset_dir, ds_name))
    dsc = load_from_disk(os.path.join(dataset_dir, ds_name+'c'))
    
    # dict to store the accuracy for each dataset
    ds_acc_dict = {}
    dsc_acc_dict = {}
    # loop for ds and dsc
    for d in [ds, dsc]:
        # loop for columns of the dataset
        for key in d.keys():
            print(f"key: {key}")
            # 予測結果の読み込み
            filename = f"{key}_pred.pkl" if d == ds else f"{ds_name}c_{key}_pred.pkl"
            with open(os.path.join(pred_out_dir, filename), "rb") as f:
                pred = pickle.load(f)
            # severityで切り出す必要がない場合
            if key in ["train", "test"] or severity == -1:
                # 予測結果のメトリクス表示
                print(f'acc: {pred.metrics["test_accuracy"]}, f1: {pred.metrics["test_f1"]}')
                if d == ds:
                    ds_acc_dict[key] = pred.metrics["test_accuracy"]["accuracy"]
                elif d == dsc:
                    dsc_acc_dict[key] = pred.metrics["test_accuracy"]["accuracy"]
            # severityで切り出す必要がある場合
            else:
                tgt_logits = get_sublist(pred.predictions, severity)
                pred_labels = np.argmax(tgt_logits, axis=-1)
                tgt_labels = get_sublist(pred.label_ids, severity)
                # accをここで計算しないといけない
                met_acc = evaluate.load("accuracy")
                met_f1 = evaluate.load("f1")
                acc = met_acc.compute(references=tgt_labels, predictions=pred_labels)
                f1 = met_f1.compute(references=tgt_labels, predictions=pred_labels, average="macro")
                # 予測結果のメトリクス表示
                print(f'acc: {acc}, f1: {f1}')
                if d == ds:
                    ds_acc_dict[key] = acc["accuracy"]
                elif d == dsc:
                    dsc_acc_dict[key] = acc["accuracy"]
    print(ds_acc_dict)
    print(dsc_acc_dict)
    # plot_accuracy(ds_acc_dict, dsc_acc_dict, severity)