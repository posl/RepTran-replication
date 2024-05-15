import os, sys, math
import numpy as np
import argparse
import pickle
from utils.constant import ViTExperiment
from utils.helper import get_corruption_types
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
    corrup_pred_dir = os.path.join(pretrained_dir, f"{ori_ct}_severity{severity}", "pred_results", "PredictionOutput")
    
    # dict to store the accuracy for each dataset
    ds_acc_dict = {}
    dsc_acc_dict = {}
    # オリジナルのデータの予測結果
    filename = f"ori_test_pred.pkl"
    with open(os.path.join(corrup_pred_dir, filename), "rb") as f:
        corrup_pred = pickle.load(f)
    # org modelのresult
    with open(os.path.join(org_pred_dir, filename), "rb") as f:
        org_pred = pickle.load(f)
    # 辞書に登録
    ds_acc_dict["test"] = org_pred.metrics["test_accuracy"]["accuracy"]
    dsc_acc_dict["test"] = corrup_pred.metrics["test_accuracy"]["accuracy"]
    # loop each noise type
    for ct in ct_list:
        filename = f"{ds_name}_{ct}_pred.pkl"
        # corrup modelのresult
        with open(os.path.join(corrup_pred_dir, filename), "rb") as f:
            corrup_pred = pickle.load(f)
        # org modelのresult
        with open(os.path.join(org_pred_dir, filename), "rb") as f:
            org_pred = pickle.load(f)
        # 辞書に登録
        ds_acc_dict[ct] = org_pred.metrics["test_accuracy"]["accuracy"]
        dsc_acc_dict[ct] = corrup_pred.metrics["test_accuracy"]["accuracy"]
    df_ds = pd.DataFrame(ds_acc_dict.items(), columns=["corruption type", "ori. acc."])
    df_dsc = pd.DataFrame(dsc_acc_dict.items(), columns=["corruption type", "ft. acc."])
    df = pd.merge(df_ds, df_dsc, on="corruption type")
    df_melted = pd.melt(df, id_vars="corruption type", value_vars=["ori. acc.", "ft. acc."], var_name="type", value_name="accuracy")
    print(df_melted)

    palette = {"ori. acc.": "skyblue", "ft. acc.": "salmon"}
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(data=df_melted, x="corruption type", y="accuracy", hue="type", palette=palette)
    # ori_ctがdf_meltで何行目かを取得
    ct_idx = df_melted[df_melted["corruption type"] == ori_ct].index[0]
    ax.patches[ct_idx].set_color("blue")
    # df_meltedの行数
    ax.patches[ct_idx].set_color("blue")
    ax.patches[ct_idx+len(df)].set_color("red")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_path = os.path.join(corrup_pred_dir, f"accuracy_per_dataset_severity{severity}.pdf")
    plt.savefig(save_path, dpi=300)
    print(f"save plot to {save_path}")