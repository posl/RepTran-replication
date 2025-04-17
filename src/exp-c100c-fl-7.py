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
from torch import nn

def get_location_path(n, fl_method, location_dir, generate_random=True, beta=None):
    if fl_method == "ours":
        assert beta is not None, "beta must be specified for 'ours' method"
        location_file = f"exp-c100c-fl-6_location_n{n}_beta{beta}_weight_ours.npy"
    elif fl_method == "bl":
        location_file = f"exp-c100c-fl-2_location_n{n}_weight_bl.npy"
    elif fl_method == "random":
        location_file = f"exp-c100c-fl-3_location_n{n}_weight_random.npy"
    else:
        raise ValueError(f"Unknown fl_method: {fl_method}")
    location_path = os.path.join(location_dir, location_file)
    return location_path

def get_loss_diff_path(n, beta, fl_method, loss_diff_dir, op, cor_mis):
    assert cor_mis in ["cor", "mis"], f"Unknown cor_mis: {cor_mis}"
    if fl_method == "ours":
        loss_diff_file = f"exp-c100c-fl-6_loss_diff_n{n}_beta{beta}_{op}_{cor_mis}_weight_ours.npy"
    elif fl_method == "bl":
        loss_diff_file = f"exp-c100c-fl-2_loss_diff_n{n}_{op}_{cor_mis}_weight_bl.npy"
    elif fl_method == "random":
        loss_diff_file = f"exp-c100c-fl-1_loss_diff_n{n}_{op}_{cor_mis}_weight_random.npy"
    else:
        raise ValueError(f"Unknown fl_method: {fl_method}")
    loss_diff_path = os.path.join(loss_diff_dir, loss_diff_file)
    return loss_diff_path

def get_output_dict(vit_from_last_layer, hs, true_labels, tgt_pos=ViTExperiment.CLS_IDX):
    # Baseline 推論
    all_logits = []
    all_proba = []
    all_pred_labels = []
    for cached_state in tqdm(hs, total=len(hs)):
        logits = vit_from_last_layer(hidden_states_before_layernorm=cached_state, tgt_pos=tgt_pos)
        proba = torch.nn.functional.softmax(logits, dim=-1)
        all_logits.append(logits.detach().cpu().numpy())
        all_proba.append(proba.detach().cpu().numpy())
    all_logits = np.concatenate(all_logits, axis=0)
    all_proba = np.concatenate(all_proba, axis=0)
    all_pred_labels = all_logits.argmax(axis=-1)
    
    # Baseline Accuracy
    is_correct = (true_labels == all_pred_labels).astype(np.int32)
    total_correct = np.sum(is_correct)
    total = len(true_labels)
    accuracy_before = total_correct / total
    print(f"Accuracy: {accuracy_before:.4f} ({total_correct}/{total})")
    
    # Baseline Loss (各サンプルごと)
    logits_tensor = torch.from_numpy(all_logits)
    labels_tensor = torch.from_numpy(true_labels)
    criterion = nn.CrossEntropyLoss(reduction="none")
    loss_all = criterion(logits_tensor, labels_tensor).detach().cpu().numpy()
    
    return {
        "all_logits": all_logits,
        "all_proba": all_proba,
        "all_pred_labels": all_pred_labels,
        "loss_all": loss_all,
    }

def main(fl_method, n, beta):
    pretrained_dir = ViTExperiment.c100.OUTPUT_DIR.format(k=0)
    pred_res_dir = os.path.join(pretrained_dir, "pred_results", "PredictionOutput")
    # デバイス (cuda, or cpu) の取得
    device = get_device()
    # このpythonのファイル名を取得
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
    # クリーンデータ (C100) のロード
    ori_ds = load_from_disk(os.path.join(ViTExperiment.DATASET_DIR, "c100_fold0"))
    ori_labels = {
        "train": np.array(ori_ds["train"][label_col]),
        "repair": np.array(ori_ds["repair"][label_col]),
        "test": np.array(ori_ds["test"][label_col])
    }
    
    # pretrained modelのロード
    model = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
    model.eval()
    vit_from_last_layer = ViTFromLastLayer(model)
    vit_from_last_layer.eval()
    # configuration
    tgt_layer = 11 # NOTE: we only use the last layer for repairing
    batch_size = ViTExperiment.BATCH_SIZE
    
    # accuracyのbottom3のノイズタイプのみ処理したい
    bottom3_keys = get_bottom3_keys_from_json(os.path.join(pretrained_dir, "corruption_accuracy.json"))
    # ノイズタイプごとの誤ったサンプルのインデックスを取得
    with open(os.path.join(pretrained_dir, "corruption_error_indices.json"), 'r') as f:
        mis_indices_dict = json.load(f)
        mis_indices_dict = {k: v for k, v in mis_indices_dict.items() if k in bottom3_keys}

    # クリーンデータで正解したサンプルのインデックスを取得
    ori_pred_labels, is_correct, indices_to_correct = get_ori_model_predictions(pred_res_dir, ori_labels, tgt_split="repair", misclf_type=None)
    
    clean_hs_save_path = os.path.join(pretrained_dir, "cache_hidden_states_before_layernorm_repair", "hidden_states_before_layernorm_11.npy")
    clean_hs = get_batched_hs(clean_hs_save_path, batch_size, indices_to_correct)
    
    # Iposに対するオリジナルモデルの出力を取得
    correct_dict = get_output_dict(vit_from_last_layer, clean_hs, ori_labels["repair"][indices_to_correct], tgt_pos=tgt_pos)
    
    op_list = ["enhance", "suppress", "multiply-2"]
        
    for rank, key in enumerate(bottom3_keys, start=1):
        print(f"Processing {key} (rank: {rank})...")
        # 対象のindices
        incorrect_indices = mis_indices_dict[key]
        corrupted_hs_save_path = os.path.join(pretrained_dir, f"cache_hidden_states_before_layernorm_{key}", "hidden_states_before_layernorm_11.npy")
        corrupted_hs = get_batched_hs(corrupted_hs_save_path, batch_size, incorrect_indices)
        # Inegに対するオリジナルモデルの出力を取得
        incorrect_dict = get_output_dict(vit_from_last_layer, corrupted_hs, labels[key][incorrect_indices], tgt_pos=tgt_pos)
        
        # 重み位置情報のロード
        location_dir = os.path.join(pretrained_dir, f"corruptions_top{rank}", "weights_location")
        location_path = get_location_path(n, fl_method, location_dir, beta=beta)
        # NOTE: only weight level here
        pos_before, pos_after = np.load(location_path, allow_pickle=True)
        print(f"pos_before.shape: {pos_before.shape}, pos_after.shape: {pos_after.shape}")
        print(f"Total number of weights: {pos_before.shape[0] + pos_after.shape[0]}")
        
        # サンプルごとのロスの保存dir
        loss_diff_dir = os.path.join(location_dir, "loss_diff_per_sample")
        os.makedirs(loss_diff_dir, exist_ok=True)
        
        op_metrics = {}
        for op in op_list:
            print(f"Processing {op} operation...")
            if "multiply" in op:
                # "multiply" を含む場合はその係数を取り出す
                op_coeff = int(op.split("multiply")[-1])
            else:
                op_coeff = op
            # opかけたモデルのロス - original modelのロスの差を保存するパス
            cor_loss_diff_path = get_loss_diff_path(n, beta, fl_method, loss_diff_dir, op, "cor")
            mis_loss_diff_path = get_loss_diff_path(n, beta, fl_method, loss_diff_dir, op, "mis")
            # モデルのコピーで初期状態から再構築
            model_copy = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
            vit_from_last_layer_mod = ViTFromLastLayer(model_copy)
            vit_from_last_layer_mod.eval()
            dummy_in = [0] * (len(pos_before) + len(pos_after))
            set_new_weights(dummy_in, pos_before, pos_after, vit_from_last_layer_mod, op=op_coeff)
            _ = vit_from_last_layer_mod(hidden_states_before_layernorm=clean_hs[0], tgt_pos=tgt_pos)
            
            # Iposへの重み操作ごのモデル出力を取得
            correct_dict_mod = get_output_dict(vit_from_last_layer_mod, clean_hs, ori_labels["repair"][indices_to_correct], tgt_pos=tgt_pos)
            # Inegへの重み操作ごのモデル出力を取得
            incorrect_dict_mod = get_output_dict(vit_from_last_layer_mod, corrupted_hs, labels[key][incorrect_indices], tgt_pos=tgt_pos)
            # 操作前後のロスの差分をIposとInegで計算
            cor_loss_diff = correct_dict_mod["loss_all"] - correct_dict["loss_all"]
            mis_loss_diff = incorrect_dict_mod["loss_all"] - incorrect_dict["loss_all"]
            print(f"cor_loss_diff.shape: {cor_loss_diff.shape}, mis_loss_diff.shape: {mis_loss_diff.shape}")
            np.save(cor_loss_diff_path, cor_loss_diff)
            np.save(mis_loss_diff_path, mis_loss_diff)
            print(f"Saved loss diff for {op} operation: {cor_loss_diff_path}, {mis_loss_diff_path}\n\n")

if __name__ == "__main__":
    results = []
    exp_list = [Experiment3, ExperimentRepair1, ExperimentRepair2]
    fl_method_list = ["ours", "bl", "random"]
    beta_list = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    
    # 全ての結果を格納するDataFrame
    all_results = pd.DataFrame()
    
    for fl_method in fl_method_list:
        # betaの設定
        if fl_method == "ours":
            bz = beta_list
        else:
            bz = [None]
        # nとwnumの設定
        if fl_method == "bl" or fl_method == "ours":
            exp_list = [Experiment1, ExperimentRepair1, ExperimentRepair2]
        else:
            exp_list = [Experiment3, ExperimentRepair1, ExperimentRepair2]
        for exp in exp_list:
            if fl_method == "bl" or fl_method == "ours":
                n_ratio = exp.NUM_IDENTIFIED_WEIGHTS
            else:
                n_ratio = exp.NUM_IDENTIFIED_NEURONS_RATIO
            for b in bz:
                print(f"\nfl_method: {fl_method}, n_ratio: {n_ratio}, beta: {b}\n===================================")
                ret_list = main(fl_method=fl_method, n=n_ratio, beta=b)
                