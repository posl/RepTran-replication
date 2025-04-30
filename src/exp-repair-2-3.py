#!/usr/bin/env python
# exp-repair-2-3.py
"""
LoRA (exp-repair-2-1.py) の予測を再評価して 1-3.py と同じスキーマで
metrics.json を更新し、CSV を生成する。
"""

import argparse, warnings, pickle, json, os, sys, re
from pathlib import Path
import numpy as np
import pandas as pd
from datasets import load_from_disk

from utils.constant import ViTExperiment
from utils.helper   import get_device, json2dict
from utils.vit_util import (
    transforms, transforms_c100, processor, ViTFromLastLayer,
    get_new_model_predictions, identfy_tgt_misclf, maybe_initialize_repair_weights_,
    get_batched_hs, get_batched_labels
)
import torch
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, DefaultDataCollator
from peft import PeftModel            # pip install peft が必要

device = get_device()
TGT_LAYER = 11


# ───────────── util ─────────────
def log_info_preds(pred_labels, true_labels, is_correct):
    print(f"pred_labels (len={len(pred_labels)}) sample: {pred_labels[:10]} ...")
    print(f"true_labels (len={len(true_labels)}) sample: {true_labels[:10]} ...")
    print(f"is_correct sample: {is_correct[:10]} ...")
    print(f"correct rate: {sum(is_correct) / len(is_correct):.4f} ({sum(is_correct)}/{len(is_correct)})")
    
def pkl_load(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)

def binary_metrics(is_old, is_new):
    acc_old   = float(is_old.mean())
    acc_new   = float(is_new.mean())
    delta_acc = acc_new - acc_old
    repair_cnt = int((~is_old &  is_new).sum())
    break_cnt  = int(( is_old & ~is_new).sum())
    repair_rate = repair_cnt / int((~is_old).sum()) if (~is_old).any() else 0.0
    break_rate  = break_cnt  / int(( is_old).sum()) if ( is_old).any() else 0.0
    return acc_old, acc_new, delta_acc, repair_rate, repair_cnt, break_rate, break_cnt

def latest_ckpt_dir(dir_):
    # checkpoint-数字 の最大値を返す
    ckpts = [p for p in dir_.iterdir() if p.is_dir() and re.match(r"checkpoint-\d+", p.name)]
    if not ckpts:
        raise FileNotFoundError("No checkpoint-* directories found")
    return max(ckpts, key=lambda p: int(p.name.split("-")[-1]))
# ────────────────────────────────

if __name__ == "__main__":
    # ============================================================== #
    # 1. CLI 引数（exp-repair-1-3.py と同じ構成に合わせる）
    # ============================================================== #
    ap = argparse.ArgumentParser()
    ap.add_argument("ds",        type=str,  choices=["c10", "c100"])
    ap.add_argument("k",         type=int,  help="fold id")
    ap.add_argument("tgt_rank",  type=int,  help="target misclassification rank")
    ap.add_argument("reps_id",   type=int,  help="repetition id")
    # ───── 同名オプション ─────
    ap.add_argument("--misclf_type", type=str, default="tgt",
                    choices=["tgt", "src_tgt", "all"])
    ap.add_argument("--fpfn",        type=str, default=None,
                    choices=["fp", "fn"])
    ap.add_argument("--alpha", type=float, default=0.5)        # 2-1 に合わせて保持
    ap.add_argument("--r",     type=int,   default=16)         # LoRA rank
    ap.add_argument("--tgt_split", type=str, default="repair",
                    choices=["repair", "test"])
    args = ap.parse_args()

    # ============================================================== #
    # 2. パス解決
    # ============================================================== #
    root = Path(getattr(ViTExperiment, args.ds).OUTPUT_DIR.format(k=args.k))
    fpfn_sfx = f"_{args.fpfn}" if (args.fpfn and args.misclf_type == "tgt") else ""
    save_dir = (
        root / (f"misclf_top{args.tgt_rank}" if args.misclf_type != "all" else "")
             / (f"{args.misclf_type}{fpfn_sfx}_repair_weight_by_de"
                if args.misclf_type != "all"
                else f"{args.misclf_type}_repair_weight_by_de")
    )
    lora_dir = save_dir / f"exp-repair-2-best_patch_r{args.r}_alpha_{args.alpha}_reps{args.reps_id}"
    pred_file = lora_dir / ( "repair_pred.pkl" if args.tgt_split=="repair"
                             else "test_pred.pkl" )
    tgt_idx_file = lora_dir / "tgt_indices.npy"   # repair 時のみ必要

    if not pred_file.exists() or (args.tgt_split=="repair" and not tgt_idx_file.exists()):
        sys.exit(f"[ERROR] required file(s) missing in {lora_dir}")

    # ============================================================== #
    # 3. データセット & 旧モデル予測
    # ============================================================== #
    ds_path = Path(ViTExperiment.DATASET_DIR) / f"{args.ds}_fold{args.k}"
    ds      = load_from_disk(ds_path)
    tf_func, label_col = (
        (transforms, "label") if args.ds=="c10" else (transforms_c100, "fine_label")
    )
    all_labels = np.array(ds[args.tgt_split][label_col])
    ds_preprocessed = ds.with_transform(tf_func)
    
    model, loading_info = ViTForImageClassification.from_pretrained(root, output_loading_info=True)
    model = model.to(device).eval()
    model = maybe_initialize_repair_weights_(model, loading_info["missing_keys"])
    vit_from_last_layer = ViTFromLastLayer(model)
    vit_from_last_layer.eval()

    TGT_LAYER = 11
    hs_save_dir = os.path.join(root, f"cache_hidden_states_before_layernorm_{args.tgt_split}")
    hs_save_path = os.path.join(hs_save_dir, f"hidden_states_before_layernorm_{TGT_LAYER}.npy")
    if not os.path.exists(hs_save_path):
        print(f"[ERROR] {hs_save_path} does not exist.")
        sys.exit(1)
    hs_before_layernorm = torch.from_numpy(np.load(hs_save_path)).to(device)

    # 全repair setの hidden states
    batch_size = ViTExperiment.BATCH_SIZE
    ori_tgt_labels = all_labels
    assert len(ori_tgt_labels) == len(hs_before_layernorm), "Mismatch between labels and hidden states"
    batch_hs_before_layernorm = get_batched_hs(hs_save_path, batch_size, device=device, hs=hs_before_layernorm)
    batch_labels = get_batched_labels(ori_tgt_labels, batch_size)
    
    # (A) 修正前: repair set 全体
    pred_labels_old, true_labels_old = get_new_model_predictions(
        vit_from_last_layer,
        batch_hs_before_layernorm,
        batch_labels,
        tgt_pos=0
    )
    is_correct_old = (pred_labels_old == true_labels_old)
    print("====== Before Patch (repair set ALL) ======")
    log_info_preds(pred_labels_old, true_labels_old, is_correct_old)
    
    # repair対象データインデックス (2-1で保存)
    misclf_info_dir = root / "misclf_info"
    misclf_pair, tgt_label, tgt_mis_indices = identfy_tgt_misclf(
        misclf_info_dir=misclf_info_dir, tgt_split=args.tgt_split, tgt_rank=args.tgt_rank, misclf_type=args.misclf_type, fpfn=args.fpfn
    )
    # repair set の誤分類ペアを識別
    if args.tgt_split == "repair":
        if not os.path.exists(tgt_idx_file):
            print(f"[ERROR] {tgt_idx_file} not found.")
            sys.exit(1)
        tgt_indices = np.load(tgt_idx_file)
    else:
        tgt_indices = []
        assert args.tgt_split == "test", f"tgt_split={args.tgt_split} is not supported."
        for idx, (pl, tl) in enumerate(zip(pred_labels_old, true_labels_old)):
            # test setを対象にする場合は，正解サンプルはtgt_indicesに入れない
            if pl == tl:
                continue
            if args.misclf_type == "src_tgt":
                if pl == misclf_pair[0] and tl == misclf_pair[1]:
                    tgt_indices.append(idx)
            elif args.misclf_type == "tgt" and args.fpfn is None:
                if pl == tgt_label or tl == tgt_label:
                    tgt_indices.append(idx)
            elif args.misclf_type == "tgt" and args.fpfn == "fp":
                if pl == tgt_label:
                    tgt_indices.append(idx)
            elif args.misclf_type == "tgt" and args.fpfn == "fn":
                if tl == tgt_label:
                    tgt_indices.append(idx)
            else:
                raise ValueError(f"misclf_type={args.misclf_type} and fpfn={args.fpfn} is not supported.")
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
    print("====== Before Patch (repair subset) ======")
    log_info_preds(pred_labels_old_tgt, true_labels_old_tgt, is_correct_old_tgt)
    
    # ============================================================== #
    # 4. 新モデル予測
    # ============================================================== #
    print("\n====== Setting LoRA Patch ======\n")
    # loraの重みをロード
    if not (lora_dir / "adapter_config.json").exists():
        adapter_path = latest_ckpt_dir(lora_dir)
    model = PeftModel.from_pretrained(model, adapter_path).to(device).eval()
    state = model.state_dict()
    scale = model.peft_config["default"].lora_alpha / model.peft_config["default"].r
    prefix = f"base_model.model.vit.encoder.layer.{str(TGT_LAYER)}.intermediate.repair"
    A = state[f"{prefix}.lora_A.default.weight"].to(device)       # (r, d)
    B = state[f"{prefix}.lora_B.default.weight"].to(device)       # (d, r)
    
    # loraの重みを使ってvit_from_last_layerのintermediate.repair.weightを更新 (元々は単位行列IだがこれをI+BAにする)
    repair_linear = vit_from_last_layer.base_model_last_layer.intermediate.repair
    with torch.no_grad():
        delta = scale * (B @ A)               # (d, d)   ＝  BA
        repair_linear.weight.data.copy_(torch.eye(delta.size(0), device=device) + delta)
    # for k, v in vit_from_last_layer.state_dict().items():
    #     if "repair" in k and not "lora" in k:
    #         print(k)
    #         print(v) # 対象レイヤだけ単位行列ではないことを確認
    
    # (C) 修正後: repair set 全体
    pred_labels_new, true_labels_new = get_new_model_predictions(
        vit_from_last_layer,
        batch_hs_before_layernorm,
        batch_labels,
        tgt_pos=0
    )
    is_correct_new = (pred_labels_new == true_labels_new)
    print("====== After Patch (repair set ALL) ======")
    log_info_preds(pred_labels_new, true_labels_new, is_correct_new)

    # (D) 修正後: 修正に使ったsubset
    pred_labels_new_tgt, true_labels_new_tgt = get_new_model_predictions(
        vit_from_last_layer,
        batch_hs_before_layernorm_tgt,
        batch_labels_tgt,
        tgt_pos=0
    )
    is_correct_new_tgt = (pred_labels_new_tgt == true_labels_new_tgt)
    print("====== After Patch (repair subset) ======")
    log_info_preds(pred_labels_new_tgt, true_labels_new_tgt, is_correct_new_tgt)

    # ============================================================== #
    # 5. 指標計算（全体 & subset）
    # ============================================================== #
    (acc_old, acc_new, delta_acc,
     repair_rate_all, repair_cnt_all,
     break_rate_all, break_cnt_all) = binary_metrics(is_correct_old, is_correct_new)

    is_old_sub = is_correct_old[tgt_indices]
    is_new_sub = is_correct_new[tgt_indices]
    (_,_,_,
     repair_rate_sub, repair_cnt_sub,
     break_rate_sub, break_cnt_sub) = binary_metrics(is_old_sub, is_new_sub)
    acc_old_sub = float(is_old_sub.mean())
    acc_new_sub = float(is_new_sub.mean())

    # ============================================================== #
    # 6. metrics.json 更新（tot_time 等があれば追記）
    # ============================================================== #
    if args.tgt_split == "repair":
        metrics_json_path = os.path.join(save_dir, f"exp-repair-2-best_patch_r{args.r}_alpha_{args.alpha}_reps{args.reps_id}.json")
        # JSONの更新: 2-1で既に tot_time などが書かれている前提
        # metrics_json_pathは存在しないといけない
        assert os.path.exists(metrics_json_path), f"{metrics_json_path} does not exist."
        metrics_dict = json2dict(metrics_json_path)
    else:
        metrics_json_path = os.path.join(save_dir, f"exp-repair-2-for_test_best_patch_r{args.r}_alpha_{args.alpha}_reps{args.reps_id}.json")
        metrics_dict = {}
    
    metrics_dict.update({
        "acc_old":               acc_old,
        "acc_new":               acc_new,
        "delta_acc":             delta_acc,
        "repair_rate_overall":   repair_rate_all,
        "repair_cnt_overall":    repair_cnt_all,
        "break_rate_overall":    break_rate_all,
        "break_cnt_overall":     break_cnt_all,
        "acc_old_tgt":           acc_old_sub,
        "acc_new_tgt":           acc_new_sub,
        "repair_rate_tgt":       repair_rate_sub,
        "repair_cnt_tgt":        repair_cnt_sub,
        "break_rate_tgt":        break_rate_sub,
        "break_cnt_tgt":         break_cnt_sub,
    })

        # 更新したmetricsを保存
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"[INFO] metrics saved => {metrics_json_path}")

    print("===== Completed exp-repair-1-3.py =====")