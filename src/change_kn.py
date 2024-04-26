import os, sys, time, math
from tqdm import tqdm
import json
import numpy as np
import torch
import argparse
from datasets import load_from_disk
from transformers import ViTForImageClassification
from utils.helper import get_device
from utils.vit_util import transforms
from utils.constant import ViTExperiment

if __name__ == "__main__":
    # プログラム引数の受け取り
    parser = argparse.ArgumentParser(description='start_layer_idx selector')
    parser.add_argument('tgt_labels', type=int, nargs='+')
    parser.add_argument('--used_column', type=str, default="train")
    parser.add_argument('--start_layer_idx', type=int, default=9)
    parser.add_argument('--tgt_method', type=str, default="ig_list")
    args = parser.parse_args()
    tgt_labels = args.tgt_labels
    start_layer_idx = args.start_layer_idx
    used_column = args.used_column
    tgt_method = args.tgt_method
    batch_size = 32
    tgt_pos = ViTExperiment.CLS_IDX # tgt_posはCLS_IDXで固定 (intermediate_statesの2次元目の0番目の要素に対応する中間層ニューロン)

    # デバイス (cuda, or cpu) の取得
    device = get_device()
    # datasetをロード (初回の読み込みだけやや時間かかる)
    cifar10 = load_from_disk(os.path.join(ViTExperiment.DATASET_DIR, "c10"))
    
    for tgt_label in tgt_labels:
        # argparseで受け取った引数のサマリーを表示
        print(f"tgt_label: {tgt_label}, start_layer_idx: {start_layer_idx}, used_column: {used_column}")

        # 指定したラベルだけ集めたデータセット
        tgt_dataset = cifar10[used_column].filter(lambda x: x["label"] == tgt_label)
        # 読み込まれた時にリアルタイムで前処理を適用するようにする
        cifar10_preprocessed = tgt_dataset.with_transform(transforms)
        # ラベルを示す文字列のlist
        labels = cifar10_preprocessed.features["label"].names
        # pretrained modelのロード
        pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR
        model = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
        model.eval()
        end_layer_idx = model.vit.config.num_hidden_layers

        # 知識ニューロンを読み込む
        save_dir = os.path.join(getattr(ViTExperiment, ds_name).OUTPUT_DIR, "neuron_scores")
        kn_path = os.path.join(save_dir, f"{tgt_method}_l{start_layer_idx}tol12_{tgt_label}.json")
        with open(kn_path, "r") as f:
            kn_dict = json.load(f)
        for op in ["enhance", "suppress"]:
            print(f"op: {op}")
            all_proba = []
            # loop for the dataset
            for data_idx, entry_dic in tqdm(enumerate(cifar10_preprocessed.iter(batch_size=batch_size)), 
                                    total=math.ceil(len(cifar10_preprocessed)/batch_size)):
                x, y = entry_dic["pixel_values"].to(device), entry_dic["labels"][0]
                # バッチに対応するhidden statesとintermediate statesの取得
                outputs = model(x, tgt_pos=tgt_pos, tgt_layer=start_layer_idx, imp_pos=kn_dict["kn"], imp_op=op)
                # outputs.logitsを確率にする
                proba = torch.nn.functional.softmax(outputs.logits, dim=-1)
                all_proba.append(proba.detach().cpu().numpy())
            all_proba = np.concatenate(all_proba, axis=0)
            # 結果をnpyで保存
            save_dir = os.path.join(getattr(ViTExperiment, ds_name).OUTPUT_DIR, "pred_results")
            save_path = os.path.join(save_dir, f"{used_column}_proba_{tgt_method}_l{start_layer_idx}tol12_{op}_{tgt_label}.npy")
            np.save(save_path, all_proba)
            print(f"proba: {all_proba.shape} is saved at {save_path}")
