"""
exp-repair-7-1-3.py  —  evaluation for exp-repair-7 (Meta3 / RQ6 neuron-score ablation)

Loads the patch produced by exp-repair-7-1-1.py and evaluates on repair/test set.
"""
import os, sys, json
import argparse
import numpy as np
import torch
from datasets import load_from_disk
from transformers import ViTForImageClassification

from utils.helper import get_device, json2dict
from utils.vit_util import (
    transforms, transforms_c100,
    ViTFromLastLayer,
    get_new_model_predictions,
    get_batched_hs,
    get_batched_labels,
    identfy_tgt_misclf,
    maybe_initialize_repair_weights_,
)
from utils.constant import ViTExperiment
from utils.log import set_exp_logging
from utils.de import set_new_weights
from logging import getLogger

logger = getLogger("base_logger")

FIXED_WNUM       = 472
ALPHA_IN_ARACHNE = 10
FIXED_ALPHA      = ALPHA_IN_ARACHNE / (1 + ALPHA_IN_ARACHNE)
FIXED_BOUNDS     = "Arachne"
TGT_LAYER        = 11


def get_dirs(pretrained_dir, tgt_rank, misclf_type, fpfn):
    misclf_ptn = f"{misclf_type}_{fpfn}" if (fpfn is not None and misclf_type == "tgt") else misclf_type
    if misclf_type == "all":
        location_save_dir = os.path.join(pretrained_dir, "all_weights_location")
        save_dir          = os.path.join(pretrained_dir, "all_repair_weight_by_de")
    else:
        location_save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_ptn}_weights_location")
        save_dir          = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_ptn}_repair_weight_by_de")
    return location_save_dir, save_dir


def log_info_preds(logger_obj, pred_labels, true_labels, is_correct):
    logger_obj.info(f"pred_labels sample: {pred_labels[:10]} ...")
    logger_obj.info(f"true_labels sample: {true_labels[:10]} ...")
    logger_obj.info(f"correct rate: {sum(is_correct) / len(is_correct):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ds",       type=str)
    parser.add_argument("k",        type=int)
    parser.add_argument("tgt_rank", type=int)
    parser.add_argument("reps_id",  type=int)
    parser.add_argument("--score_mode", type=str, required=True, choices=["vdiff", "misact"])
    parser.add_argument("--misclf_type", type=str, default="tgt", choices=["src_tgt", "tgt"])
    parser.add_argument("--fpfn",      type=str,   default=None, choices=["fp", "fn"])
    parser.add_argument("--tgt_split", type=str,   default="test", choices=["repair", "test"])
    parser.add_argument("--wnum",      type=int,   default=FIXED_WNUM,
                        help="equal weight budget N_w (default 472; use 236 for the tight-budget ablation)")
    args = parser.parse_args()

    ds_name     = args.ds
    k           = args.k
    tgt_rank    = args.tgt_rank
    reps_id     = args.reps_id
    score_mode  = args.score_mode
    misclf_type = args.misclf_type
    fpfn        = args.fpfn
    tgt_split   = args.tgt_split
    wnum        = args.wnum

    setting_id = f"n{wnum}_alpha{FIXED_ALPHA}_bounds{FIXED_BOUNDS}_{score_mode}_ours"

    pretrained_dir         = getattr(ViTExperiment, ds_name.replace("-", "_")).OUTPUT_DIR.format(k=k)
    location_save_dir, save_dir = get_dirs(pretrained_dir, tgt_rank, misclf_type, fpfn)

    location_path = os.path.join(
        location_save_dir,
        f"exp-repair-7-1-location_n{wnum}_{score_mode}_weight_ours.npy"
    )
    patch_path = os.path.join(save_dir, f"exp-repair-7-1-best_patch_{setting_id}_reps{reps_id}.npy")

    for path in (location_path, patch_path):
        if not os.path.exists(path):
            print(f"[ERROR] not found: {path}")
            sys.exit(1)

    # ── Logger ───────────────────────────────────────────────────────────────
    logger_obj = set_exp_logging(
        exp_dir=save_dir,
        exp_name=f"exp-repair-7-1-3_{setting_id}_reps{reps_id}_{tgt_split}",
    )
    logger_obj.info(f"ds={ds_name}, k={k}, tgt_rank={tgt_rank}, reps_id={reps_id}, "
                    f"score_mode={score_mode}, misclf_type={misclf_type}, fpfn={fpfn}, tgt_split={tgt_split}")

    # ── Dataset & model ──────────────────────────────────────────────────────
    if ds_name in ("c10", "tiny-imagenet"):
        tf_func   = transforms
        label_col = "label"
    elif ds_name == "c100":
        tf_func   = transforms_c100
        label_col = "fine_label"
    else:
        raise NotImplementedError(ds_name)

    ds = load_from_disk(os.path.join(ViTExperiment.DATASET_DIR, f"{ds_name}_fold{k}"))
    labels = {split: np.array(ds[split][label_col]) for split in ("train", "repair", "test")}
    device    = get_device()
    batch_size = ViTExperiment.BATCH_SIZE

    model, loading_info = ViTForImageClassification.from_pretrained(
        pretrained_dir, output_loading_info=True
    )
    model.to(device).eval()
    model = maybe_initialize_repair_weights_(model, loading_info["missing_keys"])
    vit_from_last_layer = ViTFromLastLayer(model)
    vit_from_last_layer.eval()

    # ── Load location & patch ────────────────────────────────────────────────
    pos_before, pos_after = np.load(location_path, allow_pickle=True)
    patch = np.load(patch_path, allow_pickle=True)

    # ── Hidden states for evaluation split ───────────────────────────────────
    hs_save_path = os.path.join(
        pretrained_dir,
        f"cache_hidden_states_before_layernorm_{tgt_split}",
        f"hidden_states_before_layernorm_{TGT_LAYER}.npy",
    )
    if not os.path.exists(hs_save_path):
        logger_obj.error(f"[ERROR] {hs_save_path} does not exist.")
        sys.exit(1)
    hs_tensor = torch.from_numpy(np.load(hs_save_path)).to(device)

    ori_tgt_labels = labels[tgt_split]
    batch_hs_all   = get_batched_hs(hs_save_path, batch_size, device=device, hs=hs_tensor)
    batch_labels_all = get_batched_labels(ori_tgt_labels, batch_size)

    # ── Predictions before patch ──────────────────────────────────────────────
    pred_old, true_old = get_new_model_predictions(
        vit_from_last_layer, batch_hs_all, batch_labels_all, tgt_pos=ViTExperiment.CLS_IDX,
    )
    is_correct_old = pred_old == true_old
    logger_obj.info("====== Before Patch (all) ======")
    log_info_preds(logger_obj, pred_old, true_old, is_correct_old)

    # ── Target misclassification indices in evaluation split ──────────────────
    misclf_info_dir = os.path.join(pretrained_dir, "misclf_info")
    misclf_pair, tgt_label, _ = identfy_tgt_misclf(
        misclf_info_dir, tgt_split="repair", tgt_rank=tgt_rank,
        misclf_type=misclf_type, fpfn=fpfn,
    )

    if tgt_split == "repair":
        tgt_indices_path = os.path.join(save_dir, f"exp-repair-7-1-tgt_indices_{setting_id}_reps{reps_id}.npy")
        if not os.path.exists(tgt_indices_path):
            logger_obj.error(f"[ERROR] {tgt_indices_path} not found.")
            sys.exit(1)
        tgt_indices = np.load(tgt_indices_path)
    else:
        tgt_indices = []
        for idx, (pl, tl) in enumerate(zip(pred_old, true_old)):
            if misclf_type == "src_tgt":
                if pl == misclf_pair[0] and tl == misclf_pair[1]:
                    tgt_indices.append(idx)
            elif misclf_type == "tgt" and fpfn == "fp":
                if pl == tgt_label and tl != tgt_label:
                    tgt_indices.append(idx)
            elif misclf_type == "tgt" and fpfn == "fn":
                if tl == tgt_label and pl != tgt_label:
                    tgt_indices.append(idx)
        tgt_indices = np.array(tgt_indices)

    batch_hs_tgt    = get_batched_hs(hs_save_path, batch_size, tgt_indices, device=device, hs=hs_tensor)
    batch_labels_tgt = get_batched_labels(ori_tgt_labels, batch_size, tgt_indices)
    pred_old_tgt, true_old_tgt = get_new_model_predictions(
        vit_from_last_layer, batch_hs_tgt, batch_labels_tgt, tgt_pos=ViTExperiment.CLS_IDX,
    )
    is_correct_old_tgt = pred_old_tgt == true_old_tgt
    logger_obj.info("====== Before Patch (target subset) ======")
    log_info_preds(logger_obj, pred_old_tgt, true_old_tgt, is_correct_old_tgt)

    # ── Apply patch ───────────────────────────────────────────────────────────
    set_new_weights(patch, pos_before, pos_after, vit_from_last_layer, device=device)

    pred_new, true_new = get_new_model_predictions(
        vit_from_last_layer, batch_hs_all, batch_labels_all, tgt_pos=ViTExperiment.CLS_IDX,
    )
    is_correct_new = pred_new == true_new
    logger_obj.info("====== After Patch (all) ======")
    log_info_preds(logger_obj, pred_new, true_new, is_correct_new)

    pred_new_tgt, true_new_tgt = get_new_model_predictions(
        vit_from_last_layer, batch_hs_tgt, batch_labels_tgt, tgt_pos=ViTExperiment.CLS_IDX,
    )
    is_correct_new_tgt = pred_new_tgt == true_new_tgt
    logger_obj.info("====== After Patch (target subset) ======")
    log_info_preds(logger_obj, pred_new_tgt, true_new_tgt, is_correct_new_tgt)

    # ── Metrics ───────────────────────────────────────────────────────────────
    repair_cnt  = int(np.sum(~is_correct_old     & is_correct_new))
    break_cnt   = int(np.sum( is_correct_old     & ~is_correct_new))
    repair_cnt_tgt = int(np.sum(~is_correct_old_tgt & is_correct_new_tgt))
    break_cnt_tgt  = int(np.sum( is_correct_old_tgt & ~is_correct_new_tgt))

    repair_rate = repair_cnt / max(int(np.sum(~is_correct_old)), 1)
    break_rate  = break_cnt  / max(int(np.sum( is_correct_old)), 1)
    repair_rate_tgt = repair_cnt_tgt / max(int(np.sum(~is_correct_old_tgt)), 1)
    break_rate_tgt  = break_cnt_tgt  / max(int(np.sum( is_correct_old_tgt)), 1)

    acc_old = float(np.sum(is_correct_old)) / len(is_correct_old)
    acc_new = float(np.sum(is_correct_new)) / len(is_correct_new)

    if tgt_split == "repair":
        metrics_json_path = os.path.join(
            save_dir, f"exp-repair-7-1-metrics_for_repair_{setting_id}_reps{reps_id}.json"
        )
        assert os.path.exists(metrics_json_path), f"{metrics_json_path} does not exist."
        metrics_dict = json2dict(metrics_json_path)
    else:
        metrics_json_path = os.path.join(
            save_dir, f"exp-repair-7-1-metrics_for_test_{setting_id}_reps{reps_id}.json"
        )
        metrics_dict = {}

    metrics_dict.update({
        "acc_old": acc_old,
        "acc_new": acc_new,
        "delta_acc": acc_new - acc_old,
        "repair_rate_overall": float(repair_rate),
        "repair_cnt_overall":  repair_cnt,
        "break_rate_overall":  float(break_rate),
        "break_cnt_overall":   break_cnt,
        "repair_rate_tgt": float(repair_rate_tgt),
        "repair_cnt_tgt":  repair_cnt_tgt,
        "break_rate_tgt":  float(break_rate_tgt),
        "break_cnt_tgt":   break_cnt_tgt,
    })

    if misclf_type == "src_tgt":
        slabel, tlabel = misclf_pair
        tgt_mis_old = [i for i, (pl, tl) in enumerate(zip(pred_old, true_old)) if pl == slabel and tl == tlabel]
        tgt_mis_new_cnt = sum(1 for pl, tl in zip(pred_new, true_new) if pl == slabel and tl == tlabel)
        metrics_dict["tgt_misclf_cnt_old"] = len(tgt_mis_old)
        metrics_dict["tgt_misclf_cnt_new"] = tgt_mis_new_cnt
        metrics_dict["diff_tgt_misclf_cnt"] = tgt_mis_new_cnt - len(tgt_mis_old)

    with open(metrics_json_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    logger_obj.info(f"Metrics saved => {metrics_json_path}")
    print(f"[INFO] RR={repair_rate_tgt:.4f}, BR={break_rate:.4f}")
    print(f"===== Completed exp-repair-7-1-3.py (score_mode={score_mode}, reps={reps_id}, split={tgt_split}) =====")
