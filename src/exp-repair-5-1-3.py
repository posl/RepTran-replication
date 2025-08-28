#!/usr/bin/env python
# exp-repair-1-3.py

import os, sys, time, pickle, json
import argparse
import numpy as np

import torch
from datasets import load_from_disk
from transformers import ViTForImageClassification

# Custom utility classes
from utils.helper import get_device, json2dict
from utils.vit_util import (
    transforms,
    transforms_c100,
    ViTFromLastLayer,
    get_new_model_predictions,
    get_batched_hs,
    get_batched_labels,
    identfy_tgt_misclf,
    maybe_initialize_repair_weights_
)
from utils.constant import ViTExperiment
from utils.log import set_exp_logging
from utils.de import set_new_weights
from logging import getLogger

logger = getLogger("base_logger")

############################
# Definition of constants and functions
############################

DEFAULT_SETTINGS = {
    "n": 5, 
    "num_sampled_from_correct": 200,
    "max_search_num": 50,
    "pop_size": 100,
    "alpha": 0.5
}


def log_info_preds(pred_labels, true_labels, is_correct):
    logger.info(f"pred_labels (len={len(pred_labels)}) sample: {pred_labels[:10]} ...")
    logger.info(f"true_labels (len={len(true_labels)}) sample: {true_labels[:10]} ...")
    logger.info(f"is_correct sample: {is_correct[:10]} ...")
    logger.info(f"correct rate: {sum(is_correct) / len(is_correct):.4f}")


############################
# Main processing
############################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str)
    parser.add_argument("k", type=int, help="fold id")
    parser.add_argument("tgt_rank", type=int, help="the rank of the target misclassification type")
    parser.add_argument("reps_id", type=int, help="the repetition id")
    parser.add_argument("wnum", type=int, help="the number of weights to repair")
    parser.add_argument("--setting_path", type=str, default=None)
    parser.add_argument("--fl_method", type=str, default="vdiff")
    parser.add_argument("--misclf_type", type=str, default="tgt")
    parser.add_argument("--fpfn", type=str, help="fp/fn misclassification", default=None, choices=["fp", "fn"])
    parser.add_argument("--custom_alpha", type=float, default=None)
    parser.add_argument("--custom_bounds", type=str, default=None, choices=["Arachne", "ContrRep"])
    parser.add_argument("--tgt_split", type=str, default="repair", choices=["repair", "test"])
    args = parser.parse_args()

    ds_name = args.ds
    k = args.k
    tgt_rank = args.tgt_rank
    reps_id = args.reps_id
    wnum = args.wnum
    fl_method = args.fl_method
    misclf_type = args.misclf_type
    fpfn = args.fpfn
    custom_alpha = args.custom_alpha
    custom_bounds = args.custom_bounds
    tgt_split = args.tgt_split

    logger.info(f"[INFO] ds_name={ds_name}, fold_id={k}, tgt_rank={tgt_rank}, reps_id={reps_id}, "
                f"fl_method={fl_method}, misclf_type={misclf_type}, fpfn={fpfn}, "
                f"custom_alpha={custom_alpha}, "
                f"custom_bounds={custom_bounds}, tgt_split={tgt_split}")

    #==================================================
    # 1) Determine setting_id (same as 1-1)
    #==================================================
    if args.setting_path is not None:
        if not os.path.exists(args.setting_path):
            raise FileNotFoundError(f"{args.setting_path} does not exist.")
        setting_dic = json2dict(args.setting_path)
        setting_id = os.path.basename(args.setting_path).split(".")[0].split("_")[-1]
    else:
        setting_dic = DEFAULT_SETTINGS.copy()
        setting_id = "default" if (
            custom_alpha is None and custom_bounds is None
        ) else ""
        parts = []
        if wnum is not None:
            setting_dic["wnum"] = wnum
            parts.append(f"n{wnum}")
        if custom_alpha is not None:
            setting_dic["alpha"] = custom_alpha
            parts.append(f"alpha{custom_alpha}")
        if custom_bounds is not None:
            setting_dic["bounds"] = custom_bounds
            parts.append(f"bounds{custom_bounds}")
        if parts:
            setting_id = "_".join(parts)

    #==================================================
    # 2) Build location and patch file paths
    #==================================================
    pretrained_dir = getattr(ViTExperiment, ds_name.replace("-", "_")).OUTPUT_DIR.format(k=k)

    # location_path
    if fpfn is not None and misclf_type == "tgt":
        location_save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_{fpfn}_weights_location")
    elif misclf_type == "all":
        location_save_dir = os.path.join(pretrained_dir, "all_weights_location")
    else:
        location_save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_weights_location")
    # Save file for weight position information
    if fl_method == "ours" or fl_method == "bl":
        location_filename = f"exp-repair-4-1_location_n{wnum}_weight_{fl_method}.npy"
        location_path = os.path.join(location_save_dir, location_filename)
    elif fl_method == "random":
        location_filename = f"exp-repair-4-1_location_n{wnum}_weight_random_reps{reps_id}.npy" # NOTE: Add reps_id for random (considering randomness)
        location_path = os.path.join(location_save_dir, location_filename)
    assert os.path.exists(location_path), f"{location_path} does not exist. Please run the localization phase first."
    
    # Patch file
    if fpfn is not None and misclf_type == "tgt":
        save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_{fpfn}_repair_weight_by_de")
    elif misclf_type == "all":
        save_dir = os.path.join(pretrained_dir, f"{misclf_type}_repair_weight_by_de")
    else:
        save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_repair_weight_by_de")

    patch_filename = f"exp-repair-5-1-best_patch_{setting_id}_{fl_method}_reps{reps_id}.npy"
    patch_save_path = os.path.join(save_dir, patch_filename)

    #==================================================
    # 3) Load (display shape)
    #==================================================
    if not os.path.exists(location_path):
        logger.error(f"[ERROR] location_path does not exist: {location_path}")
        sys.exit(1)
    if not os.path.exists(patch_save_path):
        logger.error(f"[ERROR] patch_save_path does not exist: {patch_save_path}")
        sys.exit(1)

    pos_before, pos_after = np.load(location_path, allow_pickle=True)
    logger.info(f"[INFO] location_path => {location_path}")
    logger.info(f" pos_before: shape={pos_before.shape}, dtype={pos_before.dtype}")
    logger.info(f" pos_after:  shape={pos_after.shape}, dtype={pos_after.dtype}")

    patch = np.load(patch_save_path, allow_pickle=True)
    logger.info(f"[INFO] patch_save_path => {patch_save_path}")
    logger.info(f" patch: shape={patch.shape}, dtype={patch.dtype}")

    #==================================================
    # 4) Load model & prediction before modification (entire repair set)
    #==================================================
    exp_name = f"exp-repair-5-1-5_{setting_id}_{fl_method}"
    logger_obj = set_exp_logging(exp_dir=save_dir, exp_name=exp_name)
    logger_obj.info("[INFO] Start evaluating patched model ...")

    device = get_device()
    ds_dirname = f"{ds_name}_fold{k}"
    ds = load_from_disk(os.path.join(ViTExperiment.DATASET_DIR, ds_dirname))

    if ds_name == "c10" or ds_name == "tiny-imagenet":
        tf_func = transforms
        label_col = "label"
    elif ds_name == "c100":
        tf_func = transforms_c100
        label_col = "fine_label"
    else:
        raise NotImplementedError(ds_name)

    labels = {
        "train": np.array(ds["train"][label_col]),
        "repair": np.array(ds["repair"][label_col]),
        "test": np.array(ds["test"][label_col])
    }
    ds_preprocessed = ds.with_transform(tf_func)

    model, loading_info = ViTForImageClassification.from_pretrained(pretrained_dir, output_loading_info=True)
    model.to(device).eval()
    model = maybe_initialize_repair_weights_(model, loading_info["missing_keys"])
    vit_from_last_layer = ViTFromLastLayer(model)
    vit_from_last_layer.eval()

    TGT_LAYER = 11
    hs_save_dir = os.path.join(pretrained_dir, f"cache_hidden_states_before_layernorm_{tgt_split}")
    hs_save_path = os.path.join(hs_save_dir, f"hidden_states_before_layernorm_{TGT_LAYER}.npy")
    if not os.path.exists(hs_save_path):
        logger_obj.error(f"[ERROR] {hs_save_path} does not exist.")
        sys.exit(1)
    hs_before_layernorm = torch.from_numpy(np.load(hs_save_path)).to(device)

    # Hidden states of entire repair set
    batch_size = ViTExperiment.BATCH_SIZE
    ori_tgt_labels = labels[tgt_split]
    batch_hs_before_layernorm = get_batched_hs(hs_save_path, batch_size, device=device, hs=hs_before_layernorm)
    batch_labels = get_batched_labels(ori_tgt_labels, batch_size)
    
    # (A) Before modification: entire repair set
    pred_labels_old, true_labels_old = get_new_model_predictions(
        vit_from_last_layer,
        batch_hs_before_layernorm,
        batch_labels,
        tgt_pos=0
    )
    is_correct_old = (pred_labels_old == true_labels_old)
    logger_obj.info("====== Before Patch (repair set ALL) ======")
    log_info_preds(pred_labels_old, true_labels_old, is_correct_old)
    
    #==================================================
    # 5) Repair target data indices (saved in 1-1)
    #==================================================
    misclf_info_dir = os.path.join(pretrained_dir, "misclf_info")
    # Identify misclassification pairs in repair set
    misclf_pair, tgt_label, tgt_mis_indices = identfy_tgt_misclf(
        misclf_info_dir,
        tgt_split="repair",
        tgt_rank=tgt_rank,
        misclf_type=misclf_type,
        fpfn=fpfn
    )
    if tgt_split == "repair":
        tgt_indices_filename = f"exp-repair-5-1-tgt_indices_{setting_id}_{fl_method}_reps{reps_id}.npy"
        tgt_indices_path = os.path.join(save_dir, tgt_indices_filename)
        if not os.path.exists(tgt_indices_path):
            logger_obj.error(f"[ERROR] {tgt_indices_path} not found.")
            sys.exit(1)
        tgt_indices = np.load(tgt_indices_path)
    else:
        tgt_indices = []
        assert tgt_split == "test", f"tgt_split={tgt_split} is not supported."
        for idx, (pl, tl) in enumerate(zip(pred_labels_old, true_labels_old)):
            if misclf_type == "src_tgt":
                if pl == misclf_pair[0] and tl == misclf_pair[1]:
                    tgt_indices.append(idx)
            elif misclf_type == "tgt" and fpfn is None:
                if pl == tgt_label or tl == tgt_label:
                    tgt_indices.append(idx)
            elif misclf_type == "tgt" and fpfn == "fp":
                if pl == tgt_label and tl != tgt_label:
                    tgt_indices.append(idx)
            elif misclf_type == "tgt" and fpfn == "fn":
                if tl == tgt_label and pl != tgt_label:
                    tgt_indices.append(idx)
            else:
                raise ValueError(f"misclf_type={misclf_type} and fpfn={fpfn} is not supported.")
        tgt_indices = np.array(tgt_indices)

    # 修正対象 (subset) の hidden states
    batch_hs_before_layernorm_tgt = get_batched_hs(hs_save_path, batch_size, tgt_indices, device=device, hs=hs_before_layernorm)
    batch_labels_tgt = get_batched_labels(ori_tgt_labels, batch_size, tgt_indices)

    # (B) 修正前: 修正対象の間違いだけを取り出したsubset
    pred_labels_old_tgt, true_labels_old_tgt = get_new_model_predictions(
        vit_from_last_layer,
        batch_hs_before_layernorm_tgt,
        batch_labels_tgt,
        tgt_pos=0
    )
    is_correct_old_tgt = (pred_labels_old_tgt == true_labels_old_tgt)
    logger_obj.info("====== Before Patch (repair subset) ======")
    log_info_preds(pred_labels_old_tgt, true_labels_old_tgt, is_correct_old_tgt)

    #==================================================
    # 6) パッチ適用 & 修正後の予測
    #==================================================
    set_new_weights(patch, pos_before, pos_after, vit_from_last_layer, device=device) # ここで vit_from_last_layer に破壊的変更される

    # (C) 修正後: repair set 全体
    pred_labels_new, true_labels_new = get_new_model_predictions(
        vit_from_last_layer,
        batch_hs_before_layernorm,
        batch_labels,
        tgt_pos=0
    )
    is_correct_new = (pred_labels_new == true_labels_new)
    logger_obj.info("====== After Patch (repair set ALL) ======")
    log_info_preds(pred_labels_new, true_labels_new, is_correct_new)

    # (D) 修正後: 修正に使ったsubset
    pred_labels_new_tgt, true_labels_new_tgt = get_new_model_predictions(
        vit_from_last_layer,
        batch_hs_before_layernorm_tgt,
        batch_labels_tgt,
        tgt_pos=0
    )
    is_correct_new_tgt = (pred_labels_new_tgt == true_labels_new_tgt)
    logger_obj.info("====== After Patch (repair subset) ======")
    log_info_preds(pred_labels_new_tgt, true_labels_new_tgt, is_correct_new_tgt)

    #==================================================
    # 7) メトリクスの計算 & 記録
    #==================================================
    # 全体acc
    acc_old = float(sum(is_correct_old)) / len(is_correct_old)
    acc_new = float(sum(is_correct_new)) / len(is_correct_new)
    delta_acc = acc_new - acc_old
    r_acc = acc_new / acc_old
    diff_correct = int(sum(is_correct_new) - sum(is_correct_old))

    # repair/break全体
    repair_cnt_overall = np.sum(~is_correct_old & is_correct_new)
    break_cnt_overall = np.sum(is_correct_old & ~is_correct_new)
    repair_rate_overall = repair_cnt_overall / np.sum(~is_correct_old) if np.sum(~is_correct_old) > 0 else 0.0
    break_rate_overall = break_cnt_overall / np.sum(is_correct_old) if np.sum(is_correct_old) > 0 else 0.0

    # subset acc
    acc_old_tgt = float(sum(is_correct_old_tgt)) / len(is_correct_old_tgt)
    acc_new_tgt = float(sum(is_correct_new_tgt)) / len(is_correct_new_tgt)
    # repair/break subset
    repair_cnt_tgt = np.sum(~is_correct_old_tgt & is_correct_new_tgt)
    break_cnt_tgt = np.sum(is_correct_old_tgt & ~is_correct_new_tgt)
    repair_rate_tgt = repair_cnt_tgt / np.sum(~is_correct_old_tgt) if np.sum(~is_correct_old_tgt) > 0 else 0.0
    break_rate_tgt = break_cnt_tgt / np.sum(is_correct_old_tgt) if np.sum(is_correct_old_tgt) > 0 else 0.0

    # JSONの更新: 1-1で既に tot_time などが書かれている前提
    if tgt_split == "repair":
        metrics_json_path = os.path.join(
            save_dir,
            f"exp-repair-5-1-metrics_for_repair_{setting_id}_{fl_method}_reps{reps_id}.json"
        )
        # metrics_json_pathは存在しないといけない
        assert os.path.exists(metrics_json_path), f"{metrics_json_path} does not exist."
        metrics_dict = json2dict(metrics_json_path)
    else:
        metrics_json_path = os.path.join(
            save_dir,
            f"exp-repair-5-1-metrics_for_{tgt_split}_{setting_id}_{fl_method}_reps{reps_id}.json"
        )
        metrics_dict = {}

    # 例と同じキーで追加
    metrics_dict["acc_old"] = acc_old
    metrics_dict["acc_new"] = acc_new
    metrics_dict["delta_acc"] = delta_acc
    metrics_dict["r_acc"] = r_acc
    metrics_dict["diff_correct"] = int(diff_correct)
    metrics_dict["repair_rate_overall"] = float(repair_rate_overall)
    metrics_dict["repair_cnt_overall"] = int(repair_cnt_overall)
    metrics_dict["break_rate_overall"] = float(break_rate_overall)
    metrics_dict["break_cnt_overall"] = int(break_cnt_overall)
    metrics_dict["repair_rate_tgt"] = float(repair_rate_tgt)
    metrics_dict["repair_cnt_tgt"] = int(repair_cnt_tgt)
    metrics_dict["break_rate_tgt"] = float(break_rate_tgt)
    metrics_dict["break_cnt_tgt"] = int(break_cnt_tgt)

    # misclf_type が src_tgt の場合の追加計測例
    if misclf_type == "src_tgt":
        logger_obj.info(f"misclf_pair={misclf_pair}, tgt_label={tgt_label}")
        slabel, tlabel = misclf_pair
        tgt_mis_indices = [] # repair setにおける頻繁な間違い方と同じものをtest setでもしていたidxをSaveするためのもの
        for idx, (pl, tl) in enumerate(zip(pred_labels_old, true_labels_old)):
            if pl == slabel and tl == tlabel:
                tgt_mis_indices.append(idx)
        tgt_misclf_cnt_old = len(tgt_mis_indices)
        # tgt_misclf_cnt: repair setで特定したターゲットの間違いの種類がtest setでどれだけあったか？
        # new_injected_faults: repair setで特定したターゲットの間違いの種類は修正後に新しく何個増えたか？
        tgt_misclf_cnt_new = 0
        new_injected_faults = 0
        for idx, (pl, tl) in enumerate(zip(pred_labels_new, true_labels_new)):
            if pl == slabel and tl == tlabel:
                tgt_misclf_cnt_new += 1
                if idx not in tgt_mis_indices:
                    new_injected_faults += 1 # repair 前と違うサンプルに対してs->tの間違い方をした
        metrics_dict["tgt_misclf_cnt_old"] = tgt_misclf_cnt_old
        metrics_dict["tgt_misclf_cnt_new"] = tgt_misclf_cnt_new
        metrics_dict["diff_tgt_misclf_cnt"] = tgt_misclf_cnt_new - tgt_misclf_cnt_old
        metrics_dict["new_injected_faults"] = new_injected_faults

    logger_obj.info(f"[INFO] metrics_dict:  {metrics_dict}")
    
    # 更新したmetricsをSave
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    logger_obj.info(f"[INFO] metrics saved => {metrics_json_path}")
    print(f"[INFO] metrics saved => {metrics_json_path}")

    logger_obj.info("===== Completed exp-repair-5-1-3.py =====")
    print("===== Completed exp-repair-5-1-3.py =====")
