import os, sys, time, json
from tqdm import tqdm
import numpy as np
import argparse
from datasets import load_from_disk
from utils.helper import get_device, get_corruption_types
from utils.vit_util import transforms, transforms_c100
from utils.constant import ViTExperiment
from venn import venn, generate_petal_labels, draw_venn, generate_colors
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # プログラム引数の受け取り
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str)
    parser.add_argument('--used_column', type=str, default="test")
    parser.add_argument('--start_layer_idx', type=int, default=9)
    parser.add_argument('--severity', type=int, help="severity of corruption (integer from 0 to 4). when set to -1, treat all as one dataset.", default=4)
    args = parser.parse_args()
    ori_ds_name = args.ds
    start_layer_idx = args.start_layer_idx
    used_column = args.used_column
    severity = args.severity
    # argparseで受け取った引数のサマリーを表示
    print(f"ori_ds_name: {ori_ds_name}, start_layer_idx: {start_layer_idx}, used_column: {used_column}, severity: {severity}")

    # Set different variables for each dataset
    if ori_ds_name == "c10":
        tf_func = transforms
        label_col = "label"
        num_labels = 10
    elif ori_ds_name == "c100":
        tf_func = transforms_c100
        label_col = "fine_label"
        num_labels = 100
    else:
        NotImplementedError

    # Get device (cuda or cpu)
    device = get_device()
    # get corruption types
    ct_list = get_corruption_types()
    # original datasetをロード
    ori_ds = load_from_disk(os.path.join(ViTExperiment.DATASET_DIR, ori_ds_name))[used_column]
    ori_labels = np.array(ori_ds[label_col])
    ori_ds = ori_ds.with_transform(tf_func)
    result_dir = os.path.join(getattr(ViTExperiment, ori_ds_name).OUTPUT_DIR, "neuron_scores")
    # 対象レイヤの設定
    start_li = start_layer_idx
    end_li = 12 # NOTE: hard coding (because dont want to load model)

    all_petal_labels = []
    tic = time.perf_counter()
    # 各ノイズに対する繰り返し
    for tgt_ct in ct_list:
        # tgt_ctに対するcorruption datasetをロード
        ds = load_from_disk(os.path.join(ViTExperiment.DATASET_DIR, f"{ori_ds_name}c_severity{severity}", tgt_ct))
        # train, testに分ける
        ds_split = ds.train_test_split(test_size=0.4, shuffle=True, seed=777)[used_column] # XXX: !SEEDは絶対固定!
        labels = np.array(ds_split[label_col])
        ct_ds = ds_split.with_transform(tf_func)

        vn_pos_dict = {}
        # 正常/ノイズデータに対するloop
        for ds_name, ds, ls in zip(["ori", "crp"], [ori_ds, ct_ds], [ori_labels, labels]):
            # 正解/不正解データに対するloop
            for cor_mis in ["cor", "mis"]:
                ds_type = f"ori_{used_column}" if ds_name == "ori" else f"{tgt_ct}_{used_column}"
                vscore_save_path = os.path.join(result_dir, f"vscore_l{start_li}tol{end_li}_all_label_{ds_type}_{cor_mis}.json")
                key = f"{ds_name}_{cor_mis}"
                with open(vscore_save_path, "r") as f:
                    vn_pos_dict[key] = json.load(f)["kn"]
                    # list of list -> set of tupleに変換 (ベン図描画のため)
                    vn_pos_dict[key] = {(int(k), int(v)) for k, v in vn_pos_dict[key]}
        # petal_labels (ベン図の各領域の要素数) の取得
        petal_labels = generate_petal_labels(vn_pos_dict.values(), fmt="{size}")
        # petal_labelsの各値をfloatの数値に直す
        petal_labels = {
            k: float(v) for k, v in petal_labels.items()
        }
        sum_petal = sum(petal_labels.values())
        all_petal_labels.append(petal_labels) # ノイズごとのpetal_labelsの合計をSave
        # petal_labelsの各値のラベルごとの平均と，全体に対する割合をラベルとしてベン図に表示
        petal_labels_for_venn = {
           k: f"{v}\n({100 * v/sum_petal:.1f}%)" for k, v in petal_labels.items()
        }
        # save venn diagram
        figsize = (6, 6)
        fig, ax = plt.subplots(figsize=figsize)
        draw_venn(
            petal_labels=petal_labels_for_venn, dataset_labels=vn_pos_dict.keys(),
            hint_hidden=False, colors=generate_colors(n_colors=len(vn_pos_dict.keys())),
            figsize=figsize, fontsize=10, legend_loc="upper right", ax=ax
        )
        ax.set_title(f"{tgt_ct.capitalize()} ({used_column})", fontsize=21)
        plt.tight_layout()
        venn_result_dir = os.path.join(result_dir, "venn")
        os.makedirs(venn_result_dir, exist_ok=True)
        venn_result_path = os.path.join(venn_result_dir, f"./{tgt_ct}_{used_column}_all_label.pdf")
        plt.savefig(venn_result_path, dpi=300)
        print(f"Venn diagram is saved at {venn_result_path}")
    # 全ノイズ平均したベン図も作りたい
    # all_petal_labelsの各要素の平均を取る
    avg_petal_labels = dict.fromkeys(all_petal_labels[0], 0)
    for petal_labels in all_petal_labels:
        avg_petal_labels = {key: avg_petal_labels[key] + petal_labels[key] for key in avg_petal_labels}
    sum_petal = sum(avg_petal_labels.values())
    avg_petal_labels_for_venn = {
        k: f"{v/len(ct_list):.1f}\n({100 * v/sum_petal:.1f}%)" for k, v in avg_petal_labels.items()
    }
    # save venn diagram
    figsize = (6, 6)
    fig, ax = plt.subplots(figsize=figsize)
    draw_venn(
        petal_labels=avg_petal_labels_for_venn, dataset_labels=vn_pos_dict.keys(),
        hint_hidden=False, colors=generate_colors(n_colors=len(vn_pos_dict.keys())),
        figsize=figsize, fontsize=10, legend_loc="upper right", ax=ax
    )
    ax.set_title(f"All corruptions ({used_column})", fontsize=21)
    plt.tight_layout()
    venn_result_dir = os.path.join(result_dir, "venn")
    os.makedirs(venn_result_dir, exist_ok=True)
    venn_result_path = os.path.join(venn_result_dir, f"./all_{used_column}_all_label.pdf")
    plt.savefig(venn_result_path, dpi=300)
    print(f"Venn diagram is saved at {venn_result_path}")

    toc = time.perf_counter()
    print(f"***** Costing time: {toc - tic:0.4f} seconds *****")
