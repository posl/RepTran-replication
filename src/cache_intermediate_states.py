import os, sys, time
import argparse
from tqdm import tqdm
import numpy as np
import torch
from datasets import load_from_disk
from transformers import ViTForImageClassification
from utils.helper import get_device
from utils.vit_util import transforms, transforms_c100
from utils.constant import ViTExperiment

if __name__ == "__main__":
    # データセットをargparseで受け取る
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str)
    args = parser.parse_args()
    ds_name = args.ds
    print(f"ds_name: {ds_name}")
    # datasetごとに違う変数のセット
    if ds_name == "c10":
        tf_func = transforms
        label_col = "label"
    elif ds_name == "c100":
        tf_func = transforms_c100
        label_col = "fine_label"
    else:
        NotImplementedError

    tgt_pos = ViTExperiment.CLS_IDX
    # デバイス (cuda, or cpu) の取得
    device = get_device()
    # datasetをロード (初回の読み込みだけやや時間かかる)
    ds = load_from_disk(os.path.join(ViTExperiment.DATASET_DIR, ds_name))
    # 読み込まれた時にリアルタイムで前処理を適用するようにする
    ds_preprocessed = ds.with_transform(tf_func)
    split_names = list(ds_preprocessed.keys()) # [train, test]
    # pretrained modelのロード
    pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR
    model = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
    model.eval()

    batch_size = ViTExperiment.BATCH_SIZE
    for split in split_names:
        print(f"Process {split}...")
        all_tgt_mid_states = []
        # loop for dataset batch
        for entry_dic in tqdm(ds_preprocessed[split].iter(batch_size=batch_size), total=len(ds_preprocessed[split])//batch_size+1):
            x, y = entry_dic["pixel_values"].to(device), entry_dic["labels"]
            output = model.forward(x, output_intermediate_states=True) # (batch_size, num_layers, num_neurons)
            # CLSトークンに対応するintermediate statesを取得
            output_mid_states = np.array([mid_states_each_layer[:, tgt_pos, :].cpu().detach().numpy()
                                                for mid_states_each_layer in output.intermediate_states])
            output_mid_states = output_mid_states.transpose(1, 0, 2) # (batch_size, num_layers, num_neurons)
            all_tgt_mid_states.append(output_mid_states)
        all_tgt_mid_states = np.concatenate(all_tgt_mid_states) # (num_samples, num_layers, num_neurons)
        
        # 各サンプルに対するレイヤごとの隠れ状態を保存していく
        # 将来的なことを考えてnumpy->tensorに変換してから保存
        num_layers = model.vit.config.num_hidden_layers
        cache_dir = os.path.join(getattr(ViTExperiment, ds_name).OUTPUT_DIR, f"cache_states_{split}")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        for l_idx in range(num_layers):
            # 特定のレイヤのstatesだけ抜き出し
            tgt_mid_states = torch.tensor(all_tgt_mid_states[:, l_idx, :]).cpu()
            # 保存
            intermediate_save_path = os.path.join(cache_dir, f"intermediate_states_l{l_idx}.pt")
            torch.save(tgt_mid_states, intermediate_save_path)
            print(f"tgt_mid_states: {tgt_mid_states.shape} is saved at {intermediate_save_path}")