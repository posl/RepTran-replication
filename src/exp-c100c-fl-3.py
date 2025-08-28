import os, sys, pickle, json, math
import shutil
from tqdm import tqdm
from collections import defaultdict
from itertools import product
import numpy as np
import pandas as pd
from transformers import ViTForImageClassification
from utils.vit_util import transforms_c100, get_batched_hs, get_batched_labels, ViTFromLastLayer, get_ori_model_predictions
from utils.constant import ViTExperiment, Experiment1, Experiment3, ExperimentRepair1, ExperimentRepair2
from utils.de import set_new_weights
from utils.helper import get_device, get_bottom3_keys_from_json
from utils.log import set_exp_logging
from logging import getLogger
from datasets import load_from_disk
import torch

def generate_random_location(n, w_num, num_mid_neurons=3072, num_ba_neurons=768):
    if type(n) is float:
        assert w_num % 8 == 0, "w_num must be a multiple of 8"
        _n = math.sqrt(w_num / 8)
        assert _n.is_integer(), "w_num must be a perfect square"
        n = int(_n)
    top_idx_dic = defaultdict(list)
    for ba in ["before", "intermediate", "after"]:
        num_neurons = num_mid_neurons if ba == "intermediate" else num_ba_neurons
        # ランダムにn もしくは 4n個のニューロンを選ぶ
        if ba == "intermediate":
            topx = 4*n
        else:
            topx = n
        top_idx_dic[ba] = np.random.choice(num_neurons, topx, replace=False)
    # before-intermediate, intermediate-afterの修正箇所を返す
    pos_before = np.array(list(product(top_idx_dic["intermediate"], top_idx_dic["before"])))
    pos_after = np.array(list(product(top_idx_dic["after"], top_idx_dic["intermediate"])))
    return pos_before, pos_after

def get_location_path(n, w_num, fl_method, location_dir, generate_random=True):
    if fl_method == "ours":
        location_file = f"exp-c100c-fl-1_location_n{n}_w{w_num}_weight.npy"
    elif fl_method == "bl":
        location_file = f"exp-c100c-fl-2_location_n{n}_weight_bl.npy"
    elif fl_method == "random":
        location_file = f"exp-c100c-fl-3_location_n{n}_weight_random.npy"
    else:
        raise ValueError(f"Unknown fl_method: {fl_method}")
    location_path = os.path.join(location_dir, location_file)
    # ランダムかつ未Saveの場合はここで計算してSaveまでやる
    if fl_method == "random" and generate_random:
        print(f"Generating random locations for {location_file}...")
        pos_before, pos_after = generate_random_location(n, w_num)
        np.save(location_path, (pos_before, pos_after))
    return location_path

def main(fl_method, n, w_num):
    pretrained_dir = ViTExperiment.c100.OUTPUT_DIR.format(k=0)
    pred_res_dir = os.path.join(pretrained_dir, "pred_results", "PredictionOutput")
    # Get device (cuda or cpu)
    device = get_device()
    # Get this Python file name
    this_file_name = os.path.basename(__file__).split(".")[0]
    logger = set_exp_logging(exp_dir=pretrained_dir, exp_name=this_file_name)
    label_col = "fine_label"
    tgt_pos = ViTExperiment.CLS_IDX
    
    # Load CIFAR-100-C dataset
    dataset_dir = ViTExperiment.DATASET_DIR
    ds = load_from_disk(os.path.join(dataset_dir, "c100c"))
    labels = {
        key: np.array(ds[key][label_col]) for key in ds.keys()
    }
    # Load clean data (C100)
    ori_ds = load_from_disk(os.path.join(ViTExperiment.DATASET_DIR, "c100_fold0"))
    ori_labels = {
        "train": np.array(ori_ds["train"][label_col]),
        "repair": np.array(ori_ds["repair"][label_col]),
        "test": np.array(ori_ds["test"][label_col])
    }
    
    # Load pretrained model
    model = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
    model.eval()
    # configuration
    tgt_layer = 11 # NOTE: we only use the last layer for repairing
    batch_size = ViTExperiment.BATCH_SIZE
    
    # Only process bottom3 noise types by accuracy
    bottom3_keys = get_bottom3_keys_from_json(os.path.join(pretrained_dir, "corruption_accuracy.json"))
    
    # ノイズタイプごとの誤ったサンプルのインデックスを取得
    with open(os.path.join(pretrained_dir, "corruption_error_indices.json"), 'r') as f:
        mis_indices_dict = json.load(f)
        mis_indices_dict = {k: v for k, v in mis_indices_dict.items() if k in bottom3_keys}

    # クリーンデータで正解したサンプルのインデックスを取得
    ori_pred_labels, is_correct, indices_to_correct = get_ori_model_predictions(pred_res_dir, ori_labels, tgt_split="repair", misclf_type=None)
    
    clean_hs_save_path = os.path.join(pretrained_dir, "cache_hidden_states_before_layernorm_repair", "hidden_states_before_layernorm_11.npy")
    clean_hs = get_batched_hs(clean_hs_save_path, batch_size, indices_to_correct)
        
    for rank, key in enumerate(bottom3_keys, start=1):
        # ラベルごとのprobaのSave先
        proba_save_dir = os.path.join(pretrained_dir, f"corruptions_top{rank}", f"exp-fl-3_proba_n{n}_w{w_num}_{fl_method}")
        os.makedirs(proba_save_dir, exist_ok=True)
        print(f"Processing {key} (rank: {rank})...")
        location_dir = os.path.join(pretrained_dir, f"corruptions_top{rank}", "weights_location")
        location_path = get_location_path(n, w_num, fl_method, location_dir)
        # NOTE: only weight level here
        pos_before, pos_after = np.load(location_path, allow_pickle=True)
        print(f"pos_before.shape: {pos_before.shape}, pos_after.shape: {pos_after.shape}")
        print(f"Total number of weights: {pos_before.shape[0] + pos_after.shape[0]}")
        # 対象のノイズタイプでオリジナルモデルが間違えたサンプルのインデックス
        tgt_mis_indices = mis_indices_dict[key]
        corrupted_hs_save_path = os.path.join(pretrained_dir, f"cache_hidden_states_before_layernorm_{key}", "hidden_states_before_layernorm_11.npy")
        corrupted_hs = get_batched_hs(corrupted_hs_save_path, batch_size, tgt_mis_indices)
        print(f"Total number of batches: {len(clean_hs) + len(corrupted_hs)} = {len(clean_hs)} + {len(corrupted_hs)}")
        
        for op in ["enhance", "suppress"]:
            vit_from_last_layer = ViTFromLastLayer(model)
            vit_from_last_layer.eval()
            # 介入を加える (重みを2倍もしくは0倍にする)
            # 重み単位での介入は，推論前に静的にやる（事前に重みを変える）
            
            
            
            # 予測の実行
            all_logits = []
            all_proba = []
            for hs in [clean_hs, corrupted_hs]:
                for cached_state in tqdm(hs, total=len(hs), desc=f"Processing {key} ({op})"):
                    # ViTFromLastLayer forward is executed here
                    logits = vit_from_last_layer(hidden_states_before_layernorm=cached_state, tgt_pos=tgt_pos)
                    proba = torch.nn.functional.softmax(logits, dim=-1)
                    logits = logits.detach().cpu().numpy()
                    proba = proba.detach().cpu().numpy()
                    all_logits.append(logits)
                    all_proba.append(proba)
            all_logits = np.concatenate(all_logits, axis=0)
            all_proba = np.concatenate(all_proba, axis=0)
            all_pred_labels = all_logits.argmax(axis=-1)
            print(f"all_logits.shape: {all_logits.shape}, all_proba.shape: {all_proba.shape}, all_pred_labels.shape: {all_pred_labels.shape}")
            
            # true_pred_labelsの値ごとにprobaを取り出す
            true_labels = np.concatenate([
                ori_labels["repair"][indices_to_correct],  # クリーンデータの正解ラベル
                labels[key][tgt_mis_indices] # ノイズタイプの誤ったサンプルの正解ラベル
            ])
            print(f"true_labels.shape: {true_labels.shape}")
            # all_logits, all_proba, all_pred_labels, true_labelsのlen()が全て同じことをassert
            assert len(all_logits) == len(all_proba) == len(all_pred_labels) == len(true_labels), f"{len(all_logits)}, {len(all_proba)}, {len(all_pred_labels)}, {len(true_labels)}"
            # Save by correct label
            proba_dict = defaultdict(list)
            for true_label, proba in zip(true_labels, all_proba):
                proba_dict[true_label].append(proba)
            for true_label, proba_list in proba_dict.items():
                proba_dict[true_label] = np.stack(proba_list)
            for true_label, proba in proba_dict.items():
                save_path = os.path.join(proba_save_dir, f"proba_{op}_{true_label}.npy")
                np.save(save_path, proba)
                print(f"saved at {save_path}")


if __name__ == "__main__":
    results = []
    exp_list = [Experiment3, ExperimentRepair1, ExperimentRepair2]
    fl_method_list = ["ours", "bl", "random"]
    
    # DataFrame to store all results
    all_results = pd.DataFrame()
    
    for fl_method in fl_method_list:
        if fl_method == "bl":
            exp_list = [Experiment1, ExperimentRepair1, ExperimentRepair2]
        else:
            exp_list = [Experiment3, ExperimentRepair1, ExperimentRepair2]
        for exp in exp_list:
            if fl_method == "bl":
                n_ratio = exp.NUM_IDENTIFIED_WEIGHTS
                w_num = None
            else:
                n_ratio, w_num = exp.NUM_IDENTIFIED_NEURONS_RATIO, exp.NUM_IDENTIFIED_WEIGHTS
                w_num = 8 * w_num * w_num
            print(f"\nfl_method: {fl_method}, n_ratio: {n_ratio}, w_num: {w_num}")
            ret_list = main(fl_method=fl_method, n=n_ratio, w_num=w_num)