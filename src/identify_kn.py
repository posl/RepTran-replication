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
    parser.add_argument("--tgt_labels", type=int, nargs="*", default=range(10))
    parser.add_argument('--used_column', type=str, default="train")
    parser.add_argument("--start_layer_idx", type=int, default=9)
    args = parser.parse_args()
    ds_name = args.ds
    tgt_labels = args.tgt_labels
    start_layer_idx = args.start_layer_idx
    used_column = args.used_column
    # argparseで受け取った引数のサマリーを表示
    print(f"tgt_labels: {tgt_labels}, start_layer_idx: {start_layer_idx}, used_column: {used_column}")

    for ig_method in ["base", "ig_list"]:
        print(f"ig_method: {ig_method}")
        for tgt_label in tgt_labels:
            print(f"tgt_label: {tgt_label}")
            # neuronごとのscoreのSaveされているnpy
            res_dir = os.path.join(getattr(ViTExperiment, ds_name).OUTPUT_DIR, "neuron_scores")
            # scoreをロード
            ig_path = os.path.join(res_dir, f"{ig_method}_l{start_layer_idx}tol12_{tgt_label}.npy")
            ig = np.load(ig_path)

            # ニューロンの特定回数を保持するcounter
            kn_counter = Counter()
            # 各サンプルに対するknowledge neuronの候補を特定
            for sample_idx, ig_sample in enumerate(ig):
                # t = 各サンプルのスコアの最大値の0.2倍 (サンプルごとのknに対する閾値)
                th_t = np.max(ig_sample) * 0.2
                # th_t以上のサンプルのインデックスのタプルを得る
                idxs = np.where(ig_sample >= th_t)
                # インデックスのタプルを1次元に変換
                t = [(l_id, n_id) for l_id, n_id in zip(idxs[0], idxs[1])]
                kn_counter.update(t)

            # 同じラベルに共通のニューロンを知識ニューロンとして特定
            p = 0.7
            kn_cnt = int(ig.shape[0] * p)
            kn = [k for k, v in kn_counter.items() if v >= kn_cnt]
            print(f"num of kn: {len(kn)}")
            # レイヤを表す部分にstart_layer_indexを足す
            kn = [(l + start_layer_idx, n) for l, n in kn]
            # numpy.int64 を組み込みの整数型に変換
            kn = [(int(l), int(n)) for l, n in kn]

            # 結果をSave
            save_dict = {}
            save_dict["num_kn"] = len(kn)
            save_dict["num_kn_per_layer"] = {l: len([n for l_id, n in kn if l_id == l]) for l in range(start_layer_idx, 12)}
            save_dict["kn"] = kn
            kn_path = os.path.join(res_dir, f"{ig_method}_l{start_layer_idx}tol12_{tgt_label}.json")
            with open(kn_path, "w") as f:
                json.dump(save_dict, f, indent=4)
            print(f"kn is saved at {kn_path}")