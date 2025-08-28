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
from utils.helper import get_device, json2dict
from utils.vit_util import transforms, transforms_c100, ViTFromLastLayer, get_new_model_predictions, get_batched_hs, get_batched_labels, identfy_tgt_misclf, get_misclf_info
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

def log_info_preds(pred_labels, true_labels, is_correct):
    logger.info(f"pred_labels (len(pred_labels)={len(pred_labels)}):\n{pred_labels}")
    logger.info(f"true_labels (len(true_labels)={len(true_labels)}):\n{true_labels}")
    logger.info(f"is_correct (len(is_correct)={len(is_correct)}):\n{is_correct}")
    logger.info(f"correct rate: {sum(is_correct) / len(is_correct):.4f}")

if __name__ == "__main__":
    # Accept dataset via argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str)
    parser.add_argument('k', type=int, help="the fold id (0 to K-1)")
    parser.add_argument('misclf_type', type=str, help="the type of misclassification (src_tgt or tgt or all)", default="tgt")
    parser.add_argument('--tgt_rank', type=int, help="the rank of the target misclassification type", default=None)
    parser.add_argument("--setting_path", type=str, help="path to the setting json file", default=None)
    parser.add_argument("--fl_method", type=str, help="the method used for FL", default="vdiff")
    parser.add_argument("--custom_n", type=int, help="the custom n for the FL", default=None)
    parser.add_argument("--custom_alpha", type=float, help="the custom alpha for the repair", default=None)
    parser.add_argument("--tgt_split", type=str, help="the split to evaluate the target misclassification type", default="test")
    parser.add_argument("--fpfn", type=str, help="the type of misclassification (fp or fn)", default=None, choices=["fp", "fn"])
    args = parser.parse_args()
    ds_name = args.ds
    k = args.k
    tgt_rank = args.tgt_rank
    setting_path = args.setting_path
    fl_method = args.fl_method
    misclf_type = args.misclf_type
    tgt_split = args.tgt_split
    custom_n = args.custom_n
    custom_alpha = args.custom_alpha
    fpfn = args.fpfn
    custom_bounds = args.custom_bounds
    print(f"ds_name: {ds_name}, fold_id: {k}, tgt_rank: {tgt_rank}, fl_method: {fl_method}, misclf_type: {misclf_type}, tgt_split: {tgt_split}, fpfn: {fpfn}")
    logger.info(f"ds_name: {ds_name}, fold_id: {k}, tgt_rank: {tgt_rank}, fl_method: {fl_method}, misclf_type: {misclf_type}, tgt_split: {tgt_split}, fpfn: {fpfn}")

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
    if fl_method == "vdiff":
        metrics_dir = os.path.join(save_dir, f"{tgt_split}_metrics_for_repair_{setting_id}.json")
    elif fl_method == "random":
        metrics_dir = os.path.join(save_dir, f"{tgt_split}_metrics_for_repair_{setting_id}_random.json")
    else:
        NotImplementedError
    
    if tgt_split == "repair":
        metrics_dic = json2dict(os.path.join(save_dir, f"metrics_for_repair_{setting_id}.json"))
        assert "tot_time" in metrics_dic, f"tot_time should be in {metrics_dir}"
    else:
        metrics_dic = {}

    # {tgt_split} setに対するhidden_states_before_layernormを取得
    hs_save_dir = os.path.join(pretrained_dir, f"cache_hidden_states_before_layernorm_{tgt_split}")
    hs_save_path = os.path.join(hs_save_dir, f"hidden_states_before_layernorm_{tgt_layer}.npy")
    assert os.path.exists(hs_save_path), f"{hs_save_path} does not exist."
    batch_hs_before_layernorm = get_batched_hs(hs_save_path, batch_size, device=device)
    batch_labels = get_batched_labels(ori_tgt_labels, batch_size)

    # FLの結果の情報をロード
    if fpfn is not None and misclf_type == "tgt":
        location_save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_{fpfn}_weights_location")
    elif misclf_type == "all":
        location_save_dir = os.path.join(pretrained_dir, f"all_weights_location")
    else:
        location_save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_weights_location")
    location_save_path = os.path.join(location_save_dir, f"location_n{setting_dic['n']}_{fl_method}.npy")
    pos_before, pos_after = np.load(location_save_path, allow_pickle=True)
    
    # DEの結果の情報をロード
    patch = np.load(patch_save_path)
    fitness_tracker = pickle.load(open(tracker_save_path, "rb"))

    # Prepare model with only the final layer
    vit_from_last_layer = ViTFromLastLayer(model)
    vit_from_last_layer.eval()

    logger.info(f"Getting original model's predictions for {tgt_split} split...")
    pred_labels_old, true_labels_old = get_new_model_predictions(vit_from_last_layer, batch_hs_before_layernorm, batch_labels, tgt_pos=0)
    is_correct_old = pred_labels_old == true_labels_old
    logger.info("Model: original, Target: all")
    log_info_preds(pred_labels_old, true_labels_old, is_correct_old)

    # 新しい重みをセット
    set_new_weights(patch=patch, model=vit_from_last_layer, pos_before=pos_before, pos_after=pos_after, device=device)

    logger.info(f"Getting repaired model's predictions for {tgt_split} split...")
    pred_labels_new, true_labels_new = get_new_model_predictions(vit_from_last_layer, batch_hs_before_layernorm, batch_labels, tgt_pos=0)
    is_correct_new = pred_labels_new == true_labels_new
    logger.info("Model: new, Target: all")
    log_info_preds(pred_labels_new, true_labels_new, is_correct_new)

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
        # misclf_typeがtgtかsrc_tgtの場合はtgt_rankが必要
        assert tgt_rank is not None, "tgt_rank is required for tgt or src_tgt misclf_type."
        misclf_info_dir = os.path.join(pretrained_dir, "misclf_info")
        # repair setに対する間違い情報を取得
        misclf_pair, tgt_label, _ = identfy_tgt_misclf(misclf_info_dir, tgt_split="repair", tgt_rank=tgt_rank, misclf_type=misclf_type, fpfn=fpfn)
        logger.info(f"misclf_pair: {misclf_pair}, tgt_label: {tgt_label}")
        # misclf_typeがsrc_tgtの場合は，slabel, tlabelを取得してその間違えかたをしたindexを取得
        if misclf_type == "src_tgt":
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
        elif misclf_type == "tgt":
            tgt_mis_indices = []
            if fpfn is None:
                used_met = "f1"
            elif fpfn == "fp":
                used_met = "precision"
            elif fpfn == "fn":
                used_met = "recall"
            for idx, (pl, tl) in enumerate(zip(pred_labels_old, true_labels_old)):
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
            for idx, (pl, tl) in enumerate(zip(pred_labels_new, true_labels_new)):
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
        repair_cnt_tgt = np.sum(~is_correct_old_tgt & is_correct_new_tgt) # s->t から t->t になった数
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

    # 修正後モデルの誤分類情報のSave
    misclf_info_dir = os.path.join(save_dir, f"misclf_info_{setting_id}")
    os.makedirs(misclf_info_dir, exist_ok=True)
    mis_matrix, mis_ranking, mis_indices, met_dict = get_misclf_info(pred_labels_new, true_labels_new, num_classes=num_classes)
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