import os, sys, time, pickle, json
import argparse
from tqdm import tqdm
import numpy as np
import torch
from datasets import load_from_disk
from transformers import ViTForImageClassification
from utils.helper import get_device, get_bottom3_keys_from_json
from utils.vit_util import transforms, transforms_c100, ViTFromLastLayer, get_vscore
from utils.constant import ViTExperiment, Experiment
from utils.log import set_exp_logging
from logging import getLogger

logger = getLogger("base_logger")


if __name__ == "__main__":
    pretrained_dir = ViTExperiment.c100.OUTPUT_DIR.format(k=0)
    # デバイス (cuda, or cpu) の取得
    device = get_device()
    # このpythonのファイル名を取得
    this_file_name = os.path.basename(__file__).split(".")[0]
    logger = set_exp_logging(exp_dir=pretrained_dir, exp_name=this_file_name)
    label_col = "fine_label"
    tgt_pos = ViTExperiment.CLS_IDX
    
    # Load CIFAR-100-C dataset
    dataset_dir = Experiment.DATASET_DIR
    ds = load_from_disk(os.path.join(dataset_dir, "c100c"))
    labels = {
        key: np.array(ds[key][label_col]) for key in ds.keys()
    }
    ds_preprocessed = ds.with_transform(transforms_c100)
    
    # pretrained modelのロード
    model = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
    model.eval()
    # configuration
    batch_size = ViTExperiment.BATCH_SIZE
    end_li = model.vit.config.num_hidden_layers
    
    # accuracyのbottom3のノイズタイプのみ処理したい
    bottom3_keys = get_bottom3_keys_from_json(os.path.join(pretrained_dir, "corruption_accuracy.json"))
    
    # ノイズタイプごとの誤ったサンプルのインデックスを取得
    with open(os.path.join(pretrained_dir, "corruption_error_indices.json"), 'r') as f:
        mis_indices_dict = json.load(f)
        mis_indices_dict = {k: v for k, v in mis_indices_dict.items() if k in bottom3_keys}
    
    for key in ds_preprocessed.keys():
        if key not in bottom3_keys:
            print(f"Skipping {key}...")
            continue
        print(f"Processing {key}...")
        # vscore保存用のディレクトリ
        vscore_dir = os.path.join(pretrained_dir, "vscores")
        # cache保存用のディレクトリ
        cache_dir = os.path.join(pretrained_dir, f"cache_states_{key}")
        os.makedirs(cache_dir, exist_ok=True)
        tgt_ds = ds_preprocessed[key].select(mis_indices_dict[key]) # 間違ったやつだけ取得
        tgt_labels = labels[key][mis_indices_dict[key]] # 間違ったやつだけ取得
        
        # loop for dataset batch
        all_mid_states = []
        all_logits = []
        for entry_dic in tqdm(tgt_ds.iter(batch_size=batch_size), total=len(tgt_ds)//batch_size+1):
            x, y = entry_dic["pixel_values"].to(device), entry_dic["labels"]
            output = model.forward(x, output_intermediate_states=True, tgt_pos=tgt_pos)
            # CLSトークンに対応するintermediate statesを取得
            output_mid_states = np.array([mid_states_each_layer[1] for mid_states_each_layer in output.intermediate_states])
            output_mid_states = output_mid_states.transpose(1, 0, 2) # (batch_size, num_layers, num_neurons)
            logits = output.logits.cpu().detach().numpy()
            all_logits.append(logits)
            all_mid_states.append(output_mid_states)
        all_logits = np.concatenate(all_logits) # (num_samples, num_labels)
        all_pred_labels = np.argmax(all_logits, axis=-1) # (num_samples, )
        all_mid_states = np.concatenate(all_mid_states) # (num_samples, num_layers, num_neurons)
        # サンプルごとの予測の正解不正解の配列を作る
        is_correct = all_pred_labels == tgt_labels # (num_samples, )
        # あっていたか否かでmid_statesを分ける
        correct_mid_states = all_mid_states[is_correct]
        incorrect_mid_states = all_mid_states[~is_correct]
        # correct_mid_statesは空のはず (=正解データは存在しない)
        assert len(correct_mid_states) == 0, "correct_mid_states should be empty."
        assert len(incorrect_mid_states) == len(tgt_ds), "incorrect_mid_states should be same length as tgt_ds."
        
        
        # 不正解データに対するVscore(Imis)の計算
        vscore_per_layer = []
        for tgt_layer in range(end_li):
            print(f"tgt_layer: {tgt_layer}")
            tgt_mid_state = incorrect_mid_states[:, tgt_layer, :] # (num_samples, num_neurons)
            # vscoreを計算
            vscore = get_vscore(tgt_mid_state)
            vscore_per_layer.append(vscore)
        vscores = np.array(vscore_per_layer) # (num_tgt_layer, num_neurons)
        # vscoresを保存
        vscore_save_path = os.path.join(vscore_dir, f"vscore_l1tol{end_li}_all_label_{key}_mis.npy")
        np.save(vscore_save_path, vscores)
        print(f"vscore ({vscores.shape}) saved at {vscore_save_path}") # mid_statesがnan (correct or incorrect 
        
        # 各サンプルに対するレイヤごとの隠れ状態を保存していく
        # 将来的なことを考えてnumpy->tensorに変換してから保存
        num_layers = model.vit.config.num_hidden_layers
        for l_idx in range(num_layers):
            # 特定のレイヤのstatesだけ抜き出し
            tgt_mid_states = torch.tensor(all_mid_states[:, l_idx, :]).cpu()
            # 保存
            intermediate_save_path = os.path.join(cache_dir, f"intermediate_states_l{l_idx}_mis.pt")
            torch.save(tgt_mid_states, intermediate_save_path)
            print(f"tgt_mid_states: {tgt_mid_states.shape} is saved at {intermediate_save_path}")