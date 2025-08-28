import os, sys, time
from tqdm import tqdm
from collections import Counter
import numpy as np
import json
import argparse
from utils.constant import ViTExperiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str)
    parser.add_argument('--used_column', type=str, default="train")
    parser.add_argument("--start_layer_idx", type=int, default=9)
    args = parser.parse_args()
    ds_name = args.ds
    start_layer_idx = args.start_layer_idx
    used_column = args.used_column
    # argparseで受け取った引数のサマリーを表示
    print(f"ds_name: {ds_name}, start_layer_idx: {start_layer_idx}, used_column: {used_column}")
    # Set different variables for each dataset
    if ds_name == "c10":
        label_col = "label"
        num_labels = 10
    elif ds_name == "c100":
        label_col = "fine_label"
        num_labels = 100
    else:
        NotImplementedError

    for tgt_label in range(num_labels):
        print(f"tgt_label: {tgt_label}")
        # neuronごとのscoreのSaveされているnpy
        res_dir = os.path.join(getattr(ViTExperiment, ds_name).OUTPUT_DIR, "neuron_scores")
        # scoreをロード
        vscore_path = os.path.join(res_dir, f"vscore_l{start_layer_idx}tol12_{tgt_label}.npy")
        vscores = np.load(vscore_path)
        # 上位3%の閾値を計算
        threshold = np.percentile(vscores.flatten(), 97)
        # 閾値以上の値を持つ要素のインデックスを取得
        indices = np.argwhere(vscores >= threshold)
        # 上位3%のインデックス (レイヤ番号, ニューロン番号) のリスト
        vn = [(int(l_id+start_layer_idx), int(n_id)) for l_id, n_id in indices]

        # 結果をSave
        save_dict = {}
        save_dict["num_kn"] = len(vn)
        save_dict["num_kn_per_layer"] = {l: len([n_id for l_id, n_id in vn if l_id == l]) for l in range(start_layer_idx, 12)}
        save_dict["kn"] = vn
        vn_path = os.path.join(res_dir, f"vscore_l{start_layer_idx}tol12_{tgt_label}.json")
        with open(vn_path, "w") as f:
            json.dump(save_dict, f, indent=4)
        print(f"vn is saved at {vn_path}")