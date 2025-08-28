import os, sys, time, pickle, json
import argparse
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")
import evaluate
from datasets import load_from_disk
from transformers import ViTForImageClassification
from utils.helper import get_device
from utils.vit_util import transforms, transforms_c100, get_ori_model_predictions, identfy_tgt_misclf, get_misclf_info
from utils.constant import ViTExperiment
from utils.log import set_exp_logging
from logging import getLogger

logger = getLogger("base_logger")
if __name__ == "__main__":
    # Accept dataset via argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str)
    parser.add_argument('k', type=int, help="the fold id (0 to K-1)")
    parser.add_argument('misclf_type', type=str, help="the type of misclassification (src_tgt or tgt or all)")
    parser.add_argument('--tgt_rank', type=int, help="the rank of the target misclassification type", default=None)
    parser.add_argument('--tgt_split', type=str, help="the split to evaluate the target misclassification type", default="test")
    parser.add_argument("--use_whole", action="store_true", help="use the whole dataset for evaluation")
    parser.add_argument("--fpfn", type=str, help="the type of misclassification (fp or fn)", default=None, choices=["fp", "fn"])
    args = parser.parse_args()
    ds_name = args.ds
    k = args.k
    tgt_rank = args.tgt_rank
    misclf_type = args.misclf_type
    tgt_split = args.tgt_split
    use_whole = args.use_whole
    fpfn = args.fpfn
    print(f"ds_name: {ds_name}, fold_id: {k}, tgt_rank: {tgt_rank}, misclf_type: {misclf_type}, tgt_split: {tgt_split}, use_whole: {use_whole}, fpfn: {fpfn}")

    ori_pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR.format(k=k)
    if use_whole:
        pretrained_dir = os.path.join(ori_pretrained_dir, "retraining_with_repair_set")
    elif fpfn is not None:
        pretrained_dir = os.path.join(ori_pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_{fpfn}_retraining_with_only_repair_target")
    else:
        pretrained_dir = os.path.join(ori_pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_retraining_with_only_repair_target")
    print(f"retrained model dir: {pretrained_dir}")
    # Create save directories for results and logs in advance
    # Get this Python file name
    this_file_name = os.path.basename(__file__).split(".")[0]
    exp_name = f"{this_file_name}"
    # Set up logger and display configuration information
    logger = set_exp_logging(exp_dir=pretrained_dir, exp_name=exp_name)
    logger.info(f"ds_name: {ds_name}, fold_id: {k}, tgt_rank: {tgt_rank}, misclf_type: {misclf_type}")

    # Set different variables for each dataset
    if ds_name == "c10":
        tf_func = transforms
        label_col = "label"
    elif ds_name == "c100":
        tf_func = transforms_c100
        label_col = "fine_label"
        num_classes = 100
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
    ori_tgt_ds = ds_preprocessed[tgt_split]
    ori_tgt_labels = labels[tgt_split]
    tgt_layer = 11 # NOTE: we only use the last layer for repairing
    logger.info(f"tgt_layer: {tgt_layer}, tgt_split: {tgt_split}")

    # repairのメトリクスをSaveするファイルのパス
    metrics_dir = os.path.join(pretrained_dir, f"{tgt_split}_metrics_for_repair_{misclf_type}.json")
    metrics_dic = {}

    # original modelsの予測結果のSaveパス
    original_pred_out_dir = os.path.join(ori_pretrained_dir, "pred_results", "PredictionOutput")
    retrained_pred_out_dir = os.path.join(pretrained_dir, "pred_results", "PredictionOutput")

    # misclf_typeがtgtかsrc_tgtの場合はtgt_rankが必要
    if misclf_type in ["tgt", "src_tgt"]:
        assert tgt_rank is not None, "tgt_rank is required for tgt or src_tgt misclf_type."
        misclf_info_dir = os.path.join(ori_pretrained_dir, "misclf_info")
        # repair setで多かった間違いの種類を取り出す
        misclf_pair, tgt_label, tgt_mis_indices = identfy_tgt_misclf(misclf_info_dir, tgt_split="repair", tgt_rank=tgt_rank, misclf_type=misclf_type, fpfn=fpfn)

    # 予測結果を取得
    if misclf_type == "tgt":
        # ori model
        pred_labels_old, is_correct_tgt, indices_to_correct_tgt, is_correct_others, indices_to_correct_others = get_ori_model_predictions(original_pred_out_dir, labels, tgt_split=tgt_split, misclf_type=misclf_type, tgt_label=tgt_label)
        is_correct_old = is_correct_tgt | is_correct_others
        indices_to_correct_old = np.concatenate([indices_to_correct_tgt, indices_to_correct_others])
        # retrained model
        pred_labels_new, is_correct_tgt, indices_to_correct_tgt, is_correct_others, indices_to_correct_others = get_ori_model_predictions(retrained_pred_out_dir, labels, tgt_split=tgt_split, misclf_type=misclf_type, tgt_label=tgt_label)
        is_correct_new = is_correct_tgt | is_correct_others
        indices_to_correct_new = np.concatenate([indices_to_correct_tgt, indices_to_correct_others])
    else: # src_tgt or all
        # ori model
        pred_labels_old, is_correct_old, indices_to_correct_old = get_ori_model_predictions(original_pred_out_dir, labels, tgt_split=tgt_split, misclf_type=misclf_type)
        # retrained model
        pred_labels_new, is_correct_new, indices_to_correct_new = get_ori_model_predictions(retrained_pred_out_dir, labels, tgt_split=tgt_split, misclf_type=misclf_type)

    # 全体的なaccの変化
    acc_old = sum(is_correct_old) / len(is_correct_old)
    acc_new = sum(is_correct_new) / len(is_correct_new)
    delta_acc = acc_new - acc_old
    logger.info(f"acc_old: {acc_old:.4f}, acc_new: {acc_new:.4f}, delta_acc: {delta_acc:.4f}")

    # repair set 全体に対するrepair, brokenを記録 (RRoverall, BRoverall)
    repair_cnt_overall = np.sum(~is_correct_old & is_correct_new)
    break_cnt_overall = np.sum(is_correct_old & ~is_correct_new)
    repair_rate_overall = repair_cnt_overall / np.sum(~is_correct_old) # 不正解 -> 正解のcnt / 不正解のcnt
    break_rate_overall = break_cnt_overall / np.sum(is_correct_old) # 正解 -> 不正解のcnt / 正解のcnt
    logger.info(f"[Overall] repair_rate: {repair_rate_overall} ({repair_cnt_overall} / {np.sum(~is_correct_old)}), break_rate: {break_rate_overall} ({break_cnt_overall} / {np.sum(is_correct_old)})")

    metrics_dic["acc_old"] = acc_old
    metrics_dic["acc_new"] = acc_new
    metrics_dic["delta_acc"] = delta_acc
    metrics_dic["repair_rate_overall"] = repair_rate_overall
    metrics_dic["repair_cnt_overall"] = int(repair_cnt_overall)
    metrics_dic["break_rate_overall"] = break_rate_overall
    metrics_dic["break_cnt_overall"] = int(break_cnt_overall)
    
    # misclf_typeがallの場合はここで終わり
    # ただ，tgtやsrc_tgtの場合は，tgt_rankの間違いに対するrepair rateも記録する
    if misclf_type == "src_tgt" or misclf_type == "tgt":
        # misclf_typeがsrc_tgtの場合は，slabel, tlabelを取得してその間違えかたをしたindexを取得
        if misclf_type == "src_tgt":
            slabel, tlabel = misclf_pair
            tgt_mis_indices = []
            for idx, (pl, tl) in enumerate(zip(pred_labels_old, labels[tgt_split])):
                if pl == slabel and tl == tlabel:
                    tgt_mis_indices.append(idx)
            tgt_misclf_cnt_old = len(tgt_mis_indices)
            # tgt_misclf_cnt: repair setで特定したターゲットの間違いの種類がtest setでどれだけあったか？
            # new_injected_faults: repair setで特定したターゲットの間違いの種類は修正後に新しく何個増えたか？
            tgt_misclf_cnt_new = 0
            new_injected_faults = 0
            for idx, (pl, tl) in enumerate(zip(pred_labels_new, labels[tgt_split])):
                if pl == slabel and tl == tlabel:
                    tgt_misclf_cnt_new += 1
                    if idx not in tgt_mis_indices:
                        new_injected_faults += 1 # repair 前と違うサンプルに対してs->tの間違い方をした
        elif misclf_type == "tgt":
            tgt_mis_indices = []
            if fpfn is None:
                used_met = "f1"
            elif fpfn == "fp":
                used_met = "precision"
            elif fpfn == "fn":
                used_met = "recall"
            for idx, (pl, tl) in enumerate(zip(pred_labels_old, labels[tgt_split])):
                if used_met == "f1":
                    cond_fpfn = (pl == tgt_label or tl == tgt_label)
                elif used_met == "precision":
                    cond_fpfn = (pl == tgt_label)
                elif used_met == "recall":
                    cond_fpfn = (tl == tgt_label)
                if cond_fpfn and pl != tl:
                    tgt_mis_indices.append(idx)
            tgt_misclf_cnt_old = len(tgt_mis_indices)
            # 修正後
            tgt_misclf_cnt_new = 0
            new_injected_faults = 0
            for idx, (pl, tl) in enumerate(zip(pred_labels_new, labels[tgt_split])):
                if used_met == "f1":
                    cond_fpfn = (pl == tgt_label or tl == tgt_label)
                elif used_met == "precision":
                    cond_fpfn = (pl == tgt_label)
                elif used_met == "recall":
                    cond_fpfn = (tl == tgt_label)
                if cond_fpfn and pl != tl:
                    tgt_misclf_cnt_new += 1
                    if idx not in tgt_mis_indices:
                        new_injected_faults += 1
            # tgt_labelに対するf1も計算してmetric_dictに追加
            metric = evaluate.load(used_met)
            metric_tgt_old = metric.compute(predictions=pred_labels_old, references=labels[tgt_split], average=None)[used_met][tgt_label]
            metric_tgt_new = metric.compute(predictions=pred_labels_new, references=labels[tgt_split], average=None)[used_met][tgt_label]
            print(metric_tgt_old, metric_tgt_new)
            metrics_dic[f"{used_met}_tgt_old"] = metric_tgt_old
            metrics_dic[f"{used_met}_tgt_new"] = metric_tgt_new
            metrics_dic[f"delta_{used_met}_tgt"] = metric_tgt_new - metric_tgt_old
        
        # tgt_mis_indicesに対するpred_labels, true_labels, is_correctを取得
        is_correct_old_tgt = is_correct_old[tgt_mis_indices]
        assert sum(is_correct_old_tgt) == 0, f"sum(is_correct_old_tgt)={sum(is_correct_old_tgt)} should be 0."
        is_correct_new_tgt = is_correct_new[tgt_mis_indices]
        # tgt_mis_indicesに対しては古いモデルは全部間違えるはず
        assert sum(is_correct_old_tgt) == 0, f"All the tgt misclf should be misclf. sum(is_correct_old_tgt)={sum(is_correct_old_tgt)}"
        repair_cnt_tgt = np.sum(~is_correct_old_tgt & is_correct_new_tgt)
        repair_rate_tgt = repair_cnt_tgt / np.sum(~is_correct_old_tgt) # 不正解 -> 正解のcnt / 不正解のcnt
        logger.info(f"[Target] repair_rate: {repair_rate_tgt} ({repair_cnt_tgt} / {np.sum(~is_correct_old_tgt)})")
        metrics_dic["repair_rate_tgt"] = repair_rate_tgt
        metrics_dic["repair_cnt_tgt"] = int(repair_cnt_tgt)
        metrics_dic["tgt_misclf_cnt_old"] = tgt_misclf_cnt_old
        metrics_dic["tgt_misclf_cnt_new"] = tgt_misclf_cnt_new
        metrics_dic["diff_tgt_misclf_cnt"] = tgt_misclf_cnt_new - tgt_misclf_cnt_old
        metrics_dic["new_injected_faults"] = new_injected_faults
        # NOTE: s->t の誤分類サンプルの予測が変わる = 治るではない．s,t以外->tになることもあり，これは誤分類タイプの変化を示す

    # metricsをSave
    logger.info(f"metrics_dic:\n{metrics_dic}")
    with open(metrics_dir, "w") as f:
        json.dump(metrics_dic, f, indent=4)
    logger.info(f"metrics are saved in {metrics_dir}")

    # 再学習後モデルの誤分類情報のSave
    misclf_info_dir = os.path.join(pretrained_dir, "misclf_info")
    os.makedirs(misclf_info_dir, exist_ok=True)
    mis_matrix, mis_ranking, mis_indices, met_dict = get_misclf_info(pred_labels_new, labels[tgt_split], num_classes=num_classes)
    np.save(os.path.join(misclf_info_dir, f"{tgt_split}_mis_matrix.npy"), mis_matrix)
    with open(os.path.join(misclf_info_dir, f"{tgt_split}_mis_ranking.pkl"), "wb") as f:
        pickle.dump(mis_ranking, f)
    with open(os.path.join(misclf_info_dir, f"{tgt_split}_mis_indices.pkl"), "wb") as f:
        pickle.dump(mis_indices, f)
    with open(os.path.join(misclf_info_dir, f"{tgt_split}_met_dict.pkl"), "wb") as f:
        pickle.dump(met_dict, f)
    print("Summary of the misclassification info:")
    print(f"mis_matrix: {mis_matrix.shape}")
    print(f"total_mis: {mis_matrix.sum()}")