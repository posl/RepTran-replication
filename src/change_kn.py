import os, sys, time, math
from tqdm import tqdm
import json
import numpy as np
import torch
import argparse
from datasets import load_from_disk
from transformers import ViTForImageClassification
from utils.helper import get_device
from utils.vit_util import transforms, transforms_c100
from utils.constant import ViTExperiment

def generate_random_positions(start_layer_idx, end_layer_idx, num_neurons, num_kn):
    """
    ランダムに知識ニューロンの位置 (start_layer_idx以上かつend_layer_idx-1以下のレイヤ番号, 0以上num_neurons以下のニューロン番号) を選ぶ
    """
    kn_list = []
    for _ in range(num_kn):
        layer_idx = np.random.randint(start_layer_idx, end_layer_idx)
        neuron_idx = np.random.randint(num_neurons)
        kn_list.append([layer_idx, neuron_idx])
    return kn_list

if __name__ == "__main__":
    # プログラム引数の受け取り
    parser = argparse.ArgumentParser(description='start_layer_idx selector')
    parser.add_argument('ds_name', type=str)
    parser.add_argument('tgt_labels', type=int, nargs='+')
    parser.add_argument('--used_column', type=str, default="train")
    parser.add_argument('--start_layer_idx', type=int, default=9)
    parser.add_argument('--tgt_method', type=str, default="ig_list")
    args = parser.parse_args()
    ds_name = args.ds_name
    tgt_labels = args.tgt_labels
    start_layer_idx = args.start_layer_idx
    used_column = args.used_column
    tgt_method = args.tgt_method
    batch_size = 32
    tgt_pos = ViTExperiment.CLS_IDX # tgt_posはCLS_IDXで固定 (intermediate_statesの2次元目の0番目の要素に対応する中間層ニューロン)

    # デバイス (cuda, or cpu) の取得
    device = get_device()
    # datasetをロード (初回の読み込みだけやや時間かかる)
    ds = load_from_disk(os.path.join(ViTExperiment.DATASET_DIR, ds_name))
    
    # datasetごとに違う変数のセット
    if ds_name == "c10" or ds_name == "c10c":
        tf_func = transforms
        label_col = "label"
    elif ds_name == "c100" or ds_name == "c100c":
        tf_func = transforms_c100
        label_col = "fine_label"
    else:
        NotImplementedError
    
    for tgt_label in tgt_labels:
        # argparseで受け取った引数のサマリーを表示
        print(f"tgt_label: {tgt_label}, start_layer_idx: {start_layer_idx}, used_column: {used_column}")

        # 指定したラベルだけ集めたデータセット
        tgt_dataset = ds[used_column].filter(lambda x: x[label_col] == tgt_label)
        # 読み込まれた時にリアルタイムで前処理を適用するようにする
        ds_preprocessed = tgt_dataset.with_transform(tf_func)
        # ラベルを示す文字列のlist
        labels = ds_preprocessed.features[label_col].names
        # pretrained modelのロード
        pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR
        model = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
        model.eval()
        end_layer_idx = model.vit.config.num_hidden_layers
        intermediate_size = model.vit.config.intermediate_size
        save_dir = os.path.join(getattr(ViTExperiment, ds_name).OUTPUT_DIR, "neuron_scores")

        # randomの場合は, vscore高いニューロンと同数をターゲットレイヤ全体からランダムに選ぶ
        if tgt_method == "random":
            vscore_kn_path = os.path.join(save_dir, f"vscore_l{start_layer_idx}tol{end_layer_idx}_{tgt_label}.json")
            with open(vscore_kn_path, "r") as f:
                vscore_kn_dict = json.load(f)
            kn_dict = dict.fromkeys(vscore_kn_dict)
            kn_dict["num_kn"] = vscore_kn_dict["num_kn"]
            kn_dict["kn"] = generate_random_positions(start_layer_idx, end_layer_idx, intermediate_size, kn_dict["num_kn"])
        else:
            # 知識ニューロンを読み込む
            kn_path = os.path.join(save_dir, f"{tgt_method}_l{start_layer_idx}tol{end_layer_idx}_{tgt_label}.json")
            with open(kn_path, "r") as f:
                kn_dict = json.load(f)
        print(f"num_of_kn: {kn_dict['num_kn']}")

        for op in ["enhance", "suppress"]:
            print(f"op: {op}")
            all_proba = []
            # loop for the dataset
            for data_idx, entry_dic in tqdm(enumerate(ds_preprocessed.iter(batch_size=batch_size)), 
                                    total=math.ceil(len(ds_preprocessed)/batch_size)):
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
