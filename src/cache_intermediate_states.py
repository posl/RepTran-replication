import os, sys, time
from tqdm import tqdm
import numpy as np
import torch
from datasets import load_from_disk
from transformers import ViTForImageClassification
from utils.helper import get_device
from utils.vit_util import transforms
from utils.constant import ViTExperiment

if __name__ == "__main__":
    tgt_pos = ViTExperiment.CLS_IDX

    # デバイス (cuda, or cpu) の取得
    device = get_device()
    # datasetをロード (初回の読み込みだけやや時間かかる)
    cifar10 = load_from_disk(os.path.join(ViTExperiment.DATASET_DIR, "c10"))
    # 読み込まれた時にリアルタイムで前処理を適用するようにする
    cifar10_preprocessed = cifar10.with_transform(transforms)
    split_names = list(cifar10_preprocessed.keys()) # [train, test]
    # pretrained modelのロード
    pretrained_dir = ViTExperiment.OUTPUT_DIR
    model = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
    model.eval()

    batch_size = ViTExperiment.BATCH_SIZE
    for split in split_names:
        print(f"Process {split}...")
        all_tgt_mid_states = []
        # loop for dataset batch
        for entry_dic in tqdm(cifar10_preprocessed[split].iter(batch_size=batch_size), total=len(cifar10_preprocessed[split])//batch_size+1):
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
        cache_dir = os.path.join(ViTExperiment.OUTPUT_DIR, f"cache_states_{split}")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        for l_idx in range(num_layers):
            # 特定のレイヤのstatesだけ抜き出し
            tgt_mid_states = torch.tensor(all_tgt_mid_states[:, l_idx, :]).cpu()
            # 保存
            intermediate_save_path = os.path.join(cache_dir, f"intermediate_states_l{l_idx}.pt")
            torch.save(tgt_mid_states, intermediate_save_path)
            print(f"tgt_mid_states: {tgt_mid_states.shape} is saved at {intermediate_save_path}")