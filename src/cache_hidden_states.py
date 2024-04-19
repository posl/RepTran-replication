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
    # ラベルを示す文字列のlist
    labels = cifar10_preprocessed["train"].features["label"].names
    # pretrained modelのロード
    pretrained_dir = ViTExperiment.OUTPUT_DIR
    model = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
    model.eval()

    batch_size = ViTExperiment.BATCH_SIZE
    all_tgt_hidden_states = []
    # loop for dataset batch
    for entry_dic in tqdm(cifar10_preprocessed["train"].iter(batch_size=batch_size), total=len(cifar10_preprocessed["train"])//batch_size+1):
        x, y = entry_dic["pixel_values"].to(device), entry_dic["labels"]
        output = model.forward(x, output_hidden_states=True)
        # CLSトークンに対応するhidden statesだけを取得
        output_hidden_states = np.array([hidden_states_each_layer[:, tgt_pos, :].cpu().detach().numpy()
                                            for hidden_states_each_layer in output.hidden_states[:-1]]) # output.hidden_statesの最後は後段のViTレイヤがないのでキャッシュする必要がない
        output_hidden_states = output_hidden_states.transpose(1, 0, 2) # (batch_size, num_layers, num_neurons)
        all_tgt_hidden_states.append(output_hidden_states)
    all_tgt_hidden_states = np.concatenate(all_tgt_hidden_states) # (num_samples, num_layers, num_neurons)
    
    # 各サンプルに対するレイヤごとの隠れ状態を保存していく
    # 将来的なことを考えてnumpy->tensorに変換してから保存
    num_layers = model.vit.config.num_hidden_layers
    cache_dir = os.path.join(ViTExperiment.OUTPUT_DIR, "cache_hidden_states")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    for l_idx in range(num_layers):
        tgt_hidden_states = torch.tensor(all_tgt_hidden_states[:, l_idx, :]).cpu()
        save_path = os.path.join(cache_dir, f"l{l_idx}.pt")
        torch.save(tgt_hidden_states, save_path)
        print(f"tgt_hidden_states: {tgt_hidden_states.shape} is saved at {save_path}")