import os, sys, time, pickle, json, math
from tqdm import tqdm
from collections import defaultdict
from itertools import product
import numpy as np
import pandas as pd
from utils.helper import get_device, json2dict
from utils.vit_util import transforms_c100
from utils.constant import ViTExperiment, Experiment1, ExperimentRepair1, ExperimentRepair2
from utils.log import set_exp_logging
from logging import getLogger
from datasets import load_from_disk
from transformers import ViTForImageClassification
import torch

def scaled_input(emb, num_points):
    """
    intermediate statesの重みを1/m (m=0,..,num_points) 倍した行列を作る
    """
    # emb: (1, ffn_size)
    baseline = torch.zeros_like(emb)  # (1, ffn_size)

    step = (emb - baseline) / num_points  # (1, ffn_size)

    res = torch.cat([torch.add(baseline, step * i) for i in range(num_points)], dim=0)  # (num_points, ffn_size)
    return res, step[0]


def main(ds_name, k, n):
    # Get device (cuda or cpu)
    device = get_device()
    # Set different variables for each dataset
    tgt_split = "repair" # NOTE: we only use repair split for repairing
    tgt_layer = 11 # NOTE: we only use the last layer for repairing
    tgt_pos = ViTExperiment.CLS_IDX
    num_points = ViTExperiment.NUM_POINTS # integrated gradientの積分近似の分割数
    
    # datasetをロード (true_labelsが欲しいので)
    ds_dirname = f"{ds_name}_fold{k}"
    ds = load_from_disk(os.path.join(ViTExperiment.DATASET_DIR, ds_dirname))
    ds_preprocessed = ds.with_transform(transforms_c100)[tgt_split]
    label_col = "fine_label"
    true_labels = ds[tgt_split][label_col]
    
    # Load pretrained model
    pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR.format(k=k)
    # location informationのSave先
    save_dir = os.path.join(pretrained_dir, f"all_weights_location")
    model = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
    model.eval()
    
    # 中間ニューロン値のキャッシュのロード
    mid_cache_dir = os.path.join(pretrained_dir, f"cache_states_{tgt_split}")
    mid_save_path = os.path.join(mid_cache_dir, f"intermediate_states_l{tgt_layer}.pt")
    cached_mid_states = torch.load(mid_save_path, map_location="cpu") # (tgt_splitのサンプル数(10000), 中間ニューロン数(3072))
    
    all_logits = []
    all_proba = []
    grad_list = []
    # 開始時刻
    ts = time.perf_counter()
    # loop for the dataset
    for data_idx, entry_dic in tqdm(enumerate(ds_preprocessed.iter(batch_size=1)), 
                            total=len(ds_preprocessed)): # NOTE: 重みを変える分でバッチ次元使ってるのでデータサンプルにバッチ次元をできない (データのバッチ化ができない)
        tgt_mid = torch.unsqueeze(cached_mid_states[data_idx], 0).to(device) # (1, 3072)
        x, y = entry_dic["pixel_values"].to(device), entry_dic["labels"][0]
        # get scaled weights
        scaled_weights, weights_step = scaled_input(tgt_mid, num_points)  # (num_points, ffn_size), (ffn_size)
        scaled_weights.requires_grad_(True)
        outputs = model(x, tgt_pos=tgt_pos, tgt_layer=tgt_layer, tmp_score=scaled_weights, tgt_label=y)
        
        # integrated gradientの計算
        grad = outputs.gradients
        # this var stores the partial diff. for each scaled weights
        grad = grad.sum(dim=0)  # (ffn_size) # ここが積分計算の近似値
        grad_list.append(grad.tolist())
    
        # 一応予測確率とかの計算
        logits = outputs.logits[-1] # outputs.logitsのshapeは (num_points, num_labels)
        proba = torch.nn.functional.softmax(logits, dim=-1)
        logits = logits.detach().cpu().numpy()
        proba = proba.detach().cpu().numpy()
        all_logits.append(logits)
        all_proba.append(proba)
    all_logits = np.concatenate(all_logits, axis=0)
    all_proba = np.concatenate(all_proba, axis=0)
    all_pred_labels = all_logits.argmax(axis=-1)
    # true_pred_labels と all_pred_labels を比較
    correct_pred = true_labels == all_pred_labels
    print(f"np.sum(correct_pred): {np.sum(correct_pred)}")
    
    grad_list = np.array(grad_list) # (num_samples, ffn_size)
    mean_grad_per_neuron = np.mean(grad_list, axis=0) # (ffn_size)
    top_neurons = np.argsort(mean_grad_per_neuron)[::-1][:n]
    places_to_fix = [[tgt_layer, pos] for pos in top_neurons]
    # places_to_fixをnpyでSave
    location_save_path = os.path.join(save_dir, f"exp-fl-2_location_n{n}_neuron_ig.npy")
    np.save(location_save_path, places_to_fix)
    print(f"saved location information to {location_save_path}")
    # 終了時刻
    te = time.perf_counter()
    elapsed_time = te - ts
    return elapsed_time
    
if __name__ == "__main__":
    ds = "c100"
    k_list = range(5)
    # n_list = [Experiment1.NUM_IDENTIFIED_NEURONS, ExperimentRepair1.NUM_IDENTIFIED_NEURONS, ExperimentRepair2.NUM_IDENTIFIED_NEURONS]
    n_list = [ExperimentRepair2.NUM_IDENTIFIED_NEURONS] # TODO REMOVE THIS LINE
    results = []
    for k, n in product(k_list, n_list):
        print(f"ds: {ds}, k: {k}, n: {n}")
        elapsed_time = main(ds, k, n)
        results.append({"ds": ds, "k": k, "n": n, "elapsed_time": elapsed_time})
    # results を csv にしてSave
    result_df = pd.DataFrame(results)
    result_df.to_csv("./exp-fl-2-1_time.csv", index=False)