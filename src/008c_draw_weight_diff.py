import os, sys, time, pickle, json
import argparse
from tqdm import tqdm
from collections import defaultdict
from itertools import product
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")
import torch
from datasets import load_from_disk
from transformers import ViTForImageClassification
from utils.helper import get_device, json2dict
from utils.vit_util import transforms, transforms_c100, ViTFromLastLayer, get_new_model_predictions, get_batched_hs, get_batched_labels, identfy_tgt_misclf
from utils.constant import ViTExperiment
from utils.log import set_exp_logging
from utils.de import set_new_weights
from logging import getLogger

logger = getLogger("base_logger")

DEFAULT_SETTINGS = {
    "n": 5, 
    "num_sampled_from_correct": 200,
    "max_search_num": 50,
    "pop_size": 100,
    "alpha": 0.5
}

def draw_weight_change(w_dict, save_dir, setting_id, all_weight=False):
    df = []
    for layer, model in product(["intermediate", "output"], ["original", "repaired", "retrained"]):
        for v in w_dict[model][layer]:
            df.append({"layer": layer, "model": model, "val": v})
    df = pd.DataFrame(df)
    plt.figure(figsize=(12, 8))
    # sns.violinplot(data=df, x="layer", y="val", hue="model", split=False, inner="quart")
    # sns.swarmplot(data=df, x="layer", y="val", hue="model", dodge=True, color=".3", size=3)
    sns.boxplot(data=df, x="layer", y="val", hue="model")
    plt.grid(True, axis='y')  # axis='y' で横軸の罫線を表示、axis='x' で縦軸は非表示
    plt.grid(False, axis='x')  # 縦軸の罫線を無効にする場合は、こちらでFalseを設定
    # x軸ラベルを消す
    plt.gca().set_xlabel("")
    if not all_weight:
        save_path = os.path.join(save_dir, f"wi_wo_distribution_{setting_id}_tgt.png")
    else:
        save_path = os.path.join(save_dir, f"wi_wo_distribution_{setting_id}_all.png")
    plt.savefig(save_path)

    # wi の old/new の distribution を比較
    for layer, model in product(["intermediate", "output"], ["original", "repaired", "retrained"]):
        wvals = w_dict[model][layer]
        logger.info(f"[layer: {layer}, model: {model}] mean={np.mean(wvals):.4f}, std={np.std(wvals):.4f}, max={np.max(wvals):.4f}, min={np.min(wvals):.4f}")

if __name__ == "__main__":
    # Accept dataset via argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str)
    parser.add_argument('k', type=int, help="the fold id (0 to K-1)")
    parser.add_argument('tgt_rank', type=int, help="the rank of the target misclassification type")
    parser.add_argument("--setting_path", type=str, help="path to the setting json file", default=None)
    parser.add_argument("--fl_method", type=str, help="the method used for FL", default="vdiff")
    parser.add_argument('--misclf_type', type=str, help="the type of misclassification (src_tgt or tgt or all)", default="tgt")
    parser.add_argument("--custom_n", type=int, help="the custom n for the FL", default=None)
    parser.add_argument("--custom_alpha", type=float, help="the custom alpha for the repair", default=None)
    parser.add_argument("--custom_bounds", type=str, help="the type of bound for the DE search space", default=None)
    parser.add_argument("--fpfn", type=str, help="the type of misclassification (fp or fn)", default=None, choices=["fp", "fn"])
    args = parser.parse_args()
    ds_name = args.ds
    k = args.k
    tgt_rank = args.tgt_rank
    setting_path = args.setting_path
    fl_method = args.fl_method
    misclf_type = args.misclf_type
    custom_n = args.custom_n
    custom_alpha = args.custom_alpha
    custom_bounds = args.custom_bounds
    fpfn = args.fpfn
    print(f"ds_name: {ds_name}, fold_id: {k}, tgt_rank: {tgt_rank}, fl_method: {fl_method}, misclf_type: {misclf_type}")
    logger.info(f"ds_name: {ds_name}, fold_id: {k}, tgt_rank: {tgt_rank}, fl_method: {fl_method}, misclf_type: {misclf_type}")

    # TODO: あとでrandomly weights selectionも実装
    if fl_method == "random":
        NotImplementedError, "randomly weights selection is not implemented yet."
    # If a settings JSON file is specified
    if setting_path is not None:
        assert os.path.exists(setting_path), f"{setting_path} does not exist."
        setting_dic = json2dict(setting_path)
        # setting_id becomes the filename in the format setting_{setting_id}.json
        setting_id = os.path.basename(setting_path).split(".")[0].split("_")[-1]
    # If no settings JSON file is specified, use custom or default settings for n and alpha only
    else:
        setting_dic = DEFAULT_SETTINGS
        # If any of custom_n, custom_alpha, custom_bounds are specified, temporarily set to empty string
        setting_id = "default" if (custom_n is None) and (custom_alpha is None) and (custom_bounds is None) else ""
        is_first = True
        if custom_n is not None:
            setting_dic["n"] = custom_n
            setting_id += f"n{custom_n}"
            is_first = False
        if custom_alpha is not None:
            setting_dic["alpha"] = custom_alpha
            setting_id += f"alpha{custom_alpha}" if is_first else f"_alpha{custom_alpha}"
            is_first = False
        if custom_bounds is not None:
            setting_dic["bounds"] = custom_bounds
            setting_id += f"bounds{custom_bounds}" if is_first else f"_bounds{custom_bounds}"
            is_first = False
    # Directory for pretrained model
    pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR.format(k=k)
    retrained_dir = os.path.join(pretrained_dir, "retraining_with_repair_set")
    # Create save directories for results and logs in advance
    if fpfn is not None and misclf_type == "tgt":
        save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_{fpfn}_repair_weight_by_de")
    elif misclf_type == "all":
        save_dir = os.path.join(pretrained_dir, f"{misclf_type}_repair_weight_by_de")
    else:
        save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_repair_weight_by_de")
    if fl_method == "vdiff":
        patch_save_path = os.path.join(save_dir, f"best_patch_{setting_id}.npy")
        tracker_save_path = os.path.join(save_dir, f"tracker_{setting_id}.pkl")
    elif fl_method == "random":
        patch_save_path = os.path.join(save_dir, f"best_patch_{setting_id}_random.npy")
        tracker_save_path = os.path.join(save_dir, f"tracker_{setting_id}_random.pkl")
    else:
        NotImplementedError
    os.makedirs(save_dir, exist_ok=True)
    # Get this Python file name
    this_file_name = os.path.basename(__file__).split(".")[0]
    exp_name = f"{this_file_name}_{setting_id}"
    # Set up logger and display configuration information
    logger = set_exp_logging(exp_dir=save_dir, exp_name=exp_name)
    logger.info(f"ds_name: {ds_name}, fold_id: {k}, setting_path: {setting_path}")
    logger.info(f"setting_dic (id={setting_id}): {setting_dic}")

    # Set different variables for each dataset
    if ds_name == "c10":
        tf_func = transforms
        label_col = "label"
    elif ds_name == "c100":
        tf_func = transforms_c100
        label_col = "fine_label"
    else:
        NotImplementedError
    tgt_pos = ViTExperiment.CLS_IDX
    ds_dirname = f"{ds_name}_fold{k}"
    # Get device (cuda or cpu)
    device = get_device()
    # Load dataset (takes some time only on first load)
    ds = load_from_disk(os.path.join(ViTExperiment.DATASET_DIR, ds_dirname))
    # Get labels (not shuffled)
    labels = {
        "train": np.array(ds["train"][label_col]),
        "repair": np.array(ds["repair"][label_col]),
        "test": np.array(ds["test"][label_col])
    }
    # Apply preprocessing in real-time when loaded
    ds_preprocessed = ds.with_transform(tf_func)
    # Load pretrained model
    model = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
    model.eval()
    end_li = model.vit.config.num_hidden_layers
    batch_size = ViTExperiment.BATCH_SIZE
    tgt_split = "repair" # NOTE: we only use repair split for repairing
    ori_tgt_ds = ds_preprocessed[tgt_split]
    ori_tgt_labels = labels[tgt_split]
    tgt_layer = 11 # NOTE: we only use the last layer for repairing
    logger.info(f"tgt_layer: {tgt_layer}, tgt_split: {tgt_split}")

    # FLの結果の情報をロード
    location_save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_weights_location")
    if misclf_type == "all":
        location_save_dir = os.path.join(pretrained_dir, f"all_weights_location")
    location_save_path = os.path.join(location_save_dir, f"location_n{setting_dic['n']}_{fl_method}.npy")
    pos_before, pos_after = np.load(location_save_path, allow_pickle=True)
    
    # DEの結果の情報をロード
    patch = np.load(patch_save_path)
    fitness_tracker = pickle.load(open(tracker_save_path, "rb"))

    # Prepare model with only the final layer
    vit_from_last_layer = ViTFromLastLayer(model)
    vit_from_last_layer.eval()

    # 変更後の対象重み
    wi_old = vit_from_last_layer.base_model_last_layer.intermediate.dense.weight.data
    wo_old = vit_from_last_layer.base_model_last_layer.output.dense.weight.data
    # torchからnumpyにしてコピー
    wi_old = wi_old.cpu().numpy().copy() # (4d, d)
    wo_old = wo_old.cpu().numpy().copy() # (d, 4d)
    
    # 新しい重みをセット
    set_new_weights(patch=patch, model=vit_from_last_layer, pos_before=pos_before, pos_after=pos_after, device=device)

    # 変更後の対象重み
    wi_new = vit_from_last_layer.base_model_last_layer.intermediate.dense.weight.data
    wo_new = vit_from_last_layer.base_model_last_layer.output.dense.weight.data
    # torchからnumpyにしてコピー
    wi_new = wi_new.cpu().numpy().copy()
    wo_new = wo_new.cpu().numpy().copy()
    # 変更された重みだけを取り出す
    wi_old_tgt = wi_old[pos_before[:, 0], pos_before[:, 1]]
    wo_old_tgt = wo_old[pos_after[:, 0], pos_after[:, 1]]
    ori_w_dict = {
        "intermediate": wi_old_tgt,
        "output": wo_old_tgt
    }
    wi_new_tgt = wi_new[pos_before[:, 0], pos_before[:, 1]]
    wo_new_tgt = wo_new[pos_after[:, 0], pos_after[:, 1]]
    new_w_dict = {
        "intermediate": wi_new_tgt,
        "output": wo_new_tgt
    }

    # retrained modelの重み
    retrained_model = ViTForImageClassification.from_pretrained(retrained_dir).to(device)
    wi_retrained = retrained_model.vit.encoder.layer[tgt_layer].intermediate.dense.weight.data
    wo_retrained = retrained_model.vit.encoder.layer[tgt_layer].output.dense.weight.data
    wi_retrained = wi_retrained.cpu().numpy().copy()
    wo_retrained = wo_retrained.cpu().numpy().copy()
    # 変更された重みだけを取り出す
    wi_retrained_tgt = wi_retrained[pos_before[:, 0], pos_before[:, 1]]
    wo_retrained_tgt = wo_retrained[pos_after[:, 0], pos_after[:, 1]]
    retrained_w_dict = {
        "intermediate": wi_retrained_tgt,
        "output": wo_retrained_tgt
    }
    w_dict = {
        "original": ori_w_dict,
        "repaired": new_w_dict,
        "retrained": retrained_w_dict
    }
    # 修正履歴の重みの値のviolin plotをSave
    draw_weight_change(w_dict, save_dir, setting_id, all_weight=False)
    # 限局した重みだけじゃなくて全重みに対しても描画
    ori_w_dict = {
        "intermediate": wi_old.flatten(),
        "output": wo_old.flatten()
    }
    new_w_dict = {
        "intermediate": wi_new.flatten(),
        "output": wo_new.flatten()
    }
    retrained_w_dict = {
        "intermediate": wi_retrained.flatten(),
        "output": wo_retrained.flatten()
    }
    w_dict = {
        "original": ori_w_dict,
        "repaired": new_w_dict,
        "retrained": retrained_w_dict
    }
    draw_weight_change(w_dict, save_dir, setting_id, all_weight=True)
    