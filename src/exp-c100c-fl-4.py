import os, sys, time, pickle, json, math
import shutil
from tqdm import tqdm
from collections import defaultdict
from itertools import product
import numpy as np
import pandas as pd
from utils.constant import ViTExperiment, Experiment1, Experiment3, ExperimentRepair1, ExperimentRepair2
from utils.log import set_exp_logging
from utils.helper import get_device, get_bottom3_keys_from_json
from utils.vit_util import transforms_c100, get_batched_hs, get_batched_labels, ViTFromLastLayer, get_ori_model_predictions
from transformers import ViTForImageClassification
from logging import getLogger
from datasets import load_from_disk
import torch

logger = getLogger("base_logger")
tgt_pos = ViTExperiment.CLS_IDX

def save_ori_model_proba(pretrained_dir, pred_res_dir, ori_labels, labels, batch_size, bottom3_keys, mis_indices_dict, vit_from_last_layer, key, ori_proba_dir):
    # クリーンデータで正解したサンプルのインデックスを取得
    ori_pred_labels, is_correct, indices_to_correct = get_ori_model_predictions(pred_res_dir, ori_labels, tgt_split="repair", misclf_type=None)
    
    clean_hs_save_path = os.path.join(pretrained_dir, "cache_hidden_states_before_layernorm_repair", "hidden_states_before_layernorm_11.npy")
    clean_hs = get_batched_hs(clean_hs_save_path, batch_size, indices_to_correct)
    
    # 対象のノイズタイプでオリジナルモデルが間違えたサンプルのインデックス
    tgt_mis_indices = mis_indices_dict[key]
    corrupted_hs_save_path = os.path.join(pretrained_dir, f"cache_hidden_states_before_layernorm_{key}", "hidden_states_before_layernorm_11.npy")
    corrupted_hs = get_batched_hs(corrupted_hs_save_path, batch_size, tgt_mis_indices)
    print(f"Total number of batches: {len(clean_hs) + len(corrupted_hs)} = {len(clean_hs)} + {len(corrupted_hs)}")
    
    # 予測の実行
    all_logits = []
    all_proba = []
    for hs in [clean_hs, corrupted_hs]:
        for cached_state in tqdm(hs, total=len(hs), desc=f"Processing {key}"):
            # ここでViTFromLastLayerのforwardが実行される
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
    # 正解ラベルごとに保存
    true_labels = np.concatenate([
        ori_labels["repair"][indices_to_correct],  # クリーンデータの正解ラベル
        labels[key][tgt_mis_indices] # ノイズタイプの誤ったサンプルの正解ラベル
    ])
    assert len(all_logits) == len(all_proba) == len(all_pred_labels) == len(true_labels), f"{len(all_logits)}, {len(all_proba)}, {len(all_pred_labels)}, {len(true_labels)}"
    # 正解ラベルごとに保存
    proba_dict = defaultdict(list)
    for true_label, proba in zip(true_labels, all_proba):
        proba_dict[true_label].append(proba)
    for true_label, proba_list in proba_dict.items():
        proba_dict[true_label] = np.stack(proba_list)
    for true_label, proba in proba_dict.items():
        try:
            save_path = os.path.join(ori_proba_dir, f"proba_{int(true_label)}.npy")
            print(f"saving to: {save_path}, shape: {proba.shape}")
            np.save(save_path, proba)
            print(f"saved at {save_path}")
        except Exception as e:
            print(f"Failed to save for label {true_label}: {e}")

def main(fl_method, n, w_num):
    pretrained_dir = ViTExperiment.c100.OUTPUT_DIR.format(k=0)
    pred_res_dir = os.path.join(pretrained_dir, "pred_results", "PredictionOutput")
    # デバイス (cuda, or cpu) の取得
    device = get_device()
    # このpythonのファイル名を取得
    this_file_name = os.path.basename(__file__).split(".")[0]
    logger = set_exp_logging(exp_dir=pretrained_dir, exp_name=this_file_name)
    label_col = "fine_label"
    label_list = range(100)
    
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
    batch_size = ViTExperiment.BATCH_SIZE
    
    # accuracyのbottom3のノイズタイプのみ処理したい
    bottom3_keys = get_bottom3_keys_from_json(os.path.join(pretrained_dir, "corruption_accuracy.json"))
    
    results = []
    
    for rank, key in enumerate(bottom3_keys, start=1):
        # ノイズタイプごとの誤ったサンプルのインデックスを取得
        with open(os.path.join(pretrained_dir, "corruption_error_indices.json"), 'r') as f:
            mis_indices_dict = json.load(f)
            mis_indices_dict = {k: v for k, v in mis_indices_dict.items() if k in bottom3_keys}
        # オリジナルのモデルの予測結果を保存するディレクトリ
        ori_proba_dir = os.path.join(pretrained_dir, f"corruptions_top{rank}", "pred_results")
        os.makedirs(ori_proba_dir, exist_ok=True)
        # 特定位置の重みを変化させた時の予測結果のディレクトリ
        proba_save_dir = os.path.join(pretrained_dir, f"corruptions_top{rank}", f"exp-fl-3_proba_n{n}_w{w_num}_{fl_method}")
        os.makedirs(proba_save_dir, exist_ok=True)
        # ori_proba_dir直下のnpyファイルの数
        npy_files = [f for f in os.listdir(ori_proba_dir) if f.endswith('.npy')]
        # 必要な場合はオリジナルモデルの予測結果を取得
        if len(npy_files) < len(label_list):
            save_ori_model_proba(pretrained_dir, pred_res_dir, ori_labels, labels, batch_size, bottom3_keys, mis_indices_dict, vit_from_last_layer, key, ori_proba_dir)
        
        for op in ["enhance", "suppress"]:
            for tl in label_list:
                print(f"op: {op}, tl: {tl}")
                ori_proba_path = os.path.join(ori_proba_dir, f"proba_{tl}.npy")
                ori_proba = np.load(ori_proba_path)[:, tl]
                # 特定位置だけopで変更したモデルの予測確率を取得
                proba_save_path = os.path.join(proba_save_dir, f"proba_{op}_{tl}.npy")
                proba = np.load(proba_save_path)[:, tl]  # tlへの予測確率
                assert np.load(ori_proba_path).shape == np.load(proba_save_path).shape, "{np.load(ori_proba_path).shape} != {np.load(proba_save_path).shape}"
                mean_diff = np.mean(proba - ori_proba)  # ラベルtlの全サンプルに対する予測確率の変化の平均
                print(f"mean_diff: {mean_diff}")
                
                results.append(
                    {
                        "n": n,
                        "num_weight": w_num if w_num is not None else 8 * n * n,
                        "fl_target": "neuron" if fl_method == "ig" else "weight",
                        "fl_method": fl_method,
                        "tgt_rank": rank,
                        "corruption": key,
                        "op": op,
                        "label": tl,
                        "diff_proba": mean_diff,
                    }
                )
    return pd.DataFrame(results)
    

if __name__ == "__main__":
    results = []
    exp_list = [Experiment3, ExperimentRepair1, ExperimentRepair2]
    fl_method_list = ["ours", "bl", "random"]
    
    # 全ての結果を格納するDataFrame
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
            result_df = main(fl_method=fl_method, n=n_ratio, w_num=w_num)
            all_results = pd.concat([all_results, result_df], ignore_index=True)
            print(f"all_results.shape: {all_results.shape}")
    # all_resultsを保存
    save_path = f"./exp-c100c-fl-4_proba_diff.csv"
    all_results.to_csv(save_path, index=False)
