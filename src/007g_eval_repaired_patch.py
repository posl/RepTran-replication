import os, sys, time, pickle, json
import argparse
from tqdm import tqdm
from collections import defaultdict
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

def draw_weight_change(wi_old_tgt, wo_old_tgt, wi_new_tgt, wo_new_tgt, save_dir, setting_id):
    # wi/wo, old/newからdfを作成
    # 列は io, on, val の3つ
    # io: wi or wo
    # on: old or new
    # val: 重みの値
    df = []
    for io, on, val in zip(["intermediate", "output", "intermediate", "output"], ["old", "old", "new", "new"], [wi_old_tgt, wo_old_tgt, wi_new_tgt, wo_new_tgt]):
        for v in val:
            df.append({"io": io, "on": on, "val": v})
    df = pd.DataFrame(df)
    plt.figure(figsize=(6, 8))
    sns.violinplot(data=df, x="io", y="val", hue="on", split=False, inner="quart")
    plt.grid(True, axis='y')  # axis='y' で横軸の罫線を表示、axis='x' で縦軸は非表示
    plt.grid(False, axis='x')  # 縦軸の罫線を無効にする場合は、こちらでFalseを設定
    # x軸ラベルを消す
    plt.gca().set_xlabel("")
    plt.savefig(os.path.join(save_dir, f"wi_wo_distribution_{setting_id}.png"))

    # wi の old/new の distribution を比較
    for tgt_name, tgt_w_old, tgt_w_new in zip(["wi", "wo"], [wi_old_tgt, wo_old_tgt], [wi_new_tgt, wo_new_tgt]):
        dist_save_path = os.path.join(save_dir, f"{tgt_name}_distribution_{setting_id}.png")
        logger.info(f"{tgt_name} old: mean={np.mean(tgt_w_old):.4f}, std={np.std(tgt_w_old):.4f}, max={np.max(tgt_w_old):.4f}, min={np.min(tgt_w_old):.4f}")
        logger.info(f"{tgt_name} new: mean={np.mean(tgt_w_new):.4f}, std={np.std(tgt_w_new):.4f}, max={np.max(tgt_w_new):.4f}, min={np.min(tgt_w_new):.4f}")

        # tgt_w_oldとtgt_w_newの (10x)%tile (xは1から10)を計算
        # percentiles_tgt_w_old = []
        # percentiles_tgt_w_new = []
        # percentile_labels = list(range(0, 101, 5))
        # for p in percentile_labels:
        #     percentiles_tgt_w_old.append(np.percentile(tgt_w_old, p))
        #     percentiles_tgt_w_new.append(np.percentile(tgt_w_new, p))
        # logger.info("percentiles_tgt_w_old", percentiles_tgt_w_old)
        # logger.info("percentiles_tgt_w_new", percentiles_tgt_w_new)

        # 横軸に重みの値，縦軸にパータイルを書いたグラフを比較
        # plt.figure(figsize=(12, 8))
        # plt.plot(percentile_labels, percentiles_tgt_w_old, label="old", color="blue", marker="o")
        # plt.plot(percentile_labels, percentiles_tgt_w_new, label="new", color="red", marker="o")
        # plt.xlabel("percentile")
        # plt.ylabel("weight value")
        # plt.legend()
        # plt.title(f"{tgt_name} weight distribution (old vs new)")
        # plt.savefig(dist_save_path)

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
    parser.add_argument('tgt_rank', type=int, help="the rank of the target misclassification type")
    parser.add_argument("--setting_path", type=str, help="path to the setting json file", default=None)
    parser.add_argument("--fl_method", type=str, help="the method used for FL", default="vdiff")
    parser.add_argument('--misclf_type', type=str, help="the type of misclassification (src_tgt or tgt or all)", default="tgt")
    parser.add_argument("--custom_n", type=int, help="the custom n for the FL", default=None)
    parser.add_argument("--custom_alpha", type=float, help="the custom alpha for the repair", default=None)
    args = parser.parse_args()
    ds_name = args.ds
    k = args.k
    tgt_rank = args.tgt_rank
    setting_path = args.setting_path
    fl_method = args.fl_method
    misclf_type = args.misclf_type
    custom_n = args.custom_n
    custom_alpha = args.custom_alpha
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
        setting_id = "default"
        if custom_n is not None:
            setting_dic["n"] = custom_n
            setting_id = f"n{custom_n}"
        if custom_alpha is not None:
            setting_dic["alpha"] = custom_alpha
            setting_id = f"alpha{custom_alpha}" if custom_n is None else f"n{custom_n}_alpha{custom_alpha}"
    # Directory for pretrained model
    pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR.format(k=k)
    # Create save directories for results and logs in advance
    save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_repair_weight_by_de")
    if misclf_type == "all":
        save_dir = os.path.join(pretrained_dir, f"all_repair_weight_by_de")
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

    # Get hidden_states_before_layernorm for repair set
    tgt_indices = np.load(os.path.join(save_dir, f"tgt_indices_{setting_id}.npy"))
    hs_save_dir = os.path.join(pretrained_dir, f"cache_hidden_states_before_layernorm_{tgt_split}")
    hs_save_path = os.path.join(hs_save_dir, f"hidden_states_before_layernorm_{tgt_layer}.npy")
    assert os.path.exists(hs_save_path), f"{hs_save_path} does not exist."
    batch_hs_before_layernorm_tgt = get_batched_hs(hs_save_path, batch_size, tgt_indices, device=device)
    batch_labels_tgt = get_batched_labels(ori_tgt_labels, batch_size, tgt_indices)
    # Also create batch for entire repair set (without specifying tgt_indices)
    batch_hs_before_layernorm = get_batched_hs(hs_save_path, batch_size, device=device)
    batch_labels = get_batched_labels(ori_tgt_labels, batch_size)

    logger.info("Predictions before the repair")
    # 修正前モデルでの予測 (entire repair set)
    pred_labels_old, true_labels_old = get_new_model_predictions(vit_from_last_layer, batch_hs_before_layernorm, batch_labels, tgt_pos=0)
    is_correct_old = pred_labels_old == true_labels_old
    logger.info("Model: old, Target: all")
    log_info_preds(pred_labels_old, true_labels_old, is_correct_old)

    # 修正前モデルでの予測 (repair に使ったデータだけ)
    pred_labels_old_tgt, true_labels_old_tgt = get_new_model_predictions(vit_from_last_layer, batch_hs_before_layernorm_tgt, batch_labels_tgt, tgt_pos=0)
    is_correct_old_tgt = pred_labels_old_tgt == true_labels_old_tgt
    logger.info("Model: old, Target: repair target")
    log_info_preds(pred_labels_old_tgt, true_labels_old_tgt, is_correct_old_tgt)
    
    # 新しい重みをセット
    set_new_weights(patch=patch, model=vit_from_last_layer, pos_before=pos_before, pos_after=pos_after, device=device)

    logger.info("Predictions after the repair")
    # 修正後モデルでの予測 (entire repair set)
    pred_labels_new, true_labels_new = get_new_model_predictions(vit_from_last_layer, batch_hs_before_layernorm, batch_labels, tgt_pos=0)
    is_correct_new = pred_labels_new == true_labels_new
    logger.info("Model: new, Target: all")
    log_info_preds(pred_labels_new, true_labels_new, is_correct_new)

    # 修正後モデルでの予測 (repair に使ったデータだけ)
    pred_labels_new_tgt, true_labels_new_tgt = get_new_model_predictions(vit_from_last_layer, batch_hs_before_layernorm_tgt, batch_labels_tgt, tgt_pos=0)
    is_correct_new_tgt = pred_labels_new_tgt == true_labels_new_tgt
    logger.info("Model: new, Target: repair target")
    log_info_preds(pred_labels_new_tgt, true_labels_new_tgt, is_correct_new_tgt)

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

    # repair target だけに対するrepair, brokenを記録 (RRtgt, BRtgt)
    repair_cnt_tgt = np.sum(~is_correct_old_tgt & is_correct_new_tgt)
    break_cnt_tgt = np.sum(is_correct_old_tgt & ~is_correct_new_tgt)
    repair_rate_tgt = repair_cnt_tgt / np.sum(~is_correct_old_tgt) # 不正解 -> 正解のcnt / 不正解のcnt
    break_rate_tgt = break_cnt_tgt / np.sum(is_correct_old_tgt) # 正解 -> 不正解のcnt / 正解のcnt
    logger.info(f"[Target] repair_rate: {repair_rate_tgt} ({repair_cnt_tgt} / {np.sum(~is_correct_old_tgt)}), break_rate: {break_rate_tgt} ({break_cnt_tgt} / {np.sum(is_correct_old_tgt)})")

    # cnt系の変数は全てint()する (np.intのままだとjson.dumpでこけるため)
    repair_cnt_overall = int(repair_cnt_overall)
    break_cnt_overall = int(break_cnt_overall)
    repair_cnt_tgt = int(repair_cnt_tgt)
    break_cnt_tgt = int(break_cnt_tgt)

    # 上記のメトリクスをjsonでSave
    # 007eの段階でtot_timeだけはjsonにSaveされている前提
    if fl_method == "vdiff":
        metrics_dir = os.path.join(save_dir, f"metrics_for_repair_{setting_id}.json")
    elif fl_method == "random":
        metrics_dir = os.path.join(save_dir, f"metrics_for_repair_{setting_id}_random.json")
    else:
        NotImplementedError
    metrics_dic = json2dict(metrics_dir)
    assert "tot_time" in metrics_dic, f"tot_time should be in {metrics_dir}"
    # tot_time以外のmetricを追加
    metrics_dic["acc_old"] = acc_old
    metrics_dic["acc_new"] = acc_new
    metrics_dic["delta_acc"] = delta_acc
    metrics_dic["repair_rate_overall"] = repair_rate_overall
    metrics_dic["repair_cnt_overall"] = repair_cnt_overall
    metrics_dic["break_rate_overall"] = break_rate_overall
    metrics_dic["break_cnt_overall"] = break_cnt_overall
    metrics_dic["repair_rate_tgt"] = repair_rate_tgt
    metrics_dic["repair_cnt_tgt"] = repair_cnt_tgt
    metrics_dic["break_rate_tgt"] = break_rate_tgt
    metrics_dic["break_cnt_tgt"] = break_cnt_tgt

    if misclf_type == "src_tgt":
        misclf_info_dir = os.path.join(pretrained_dir, "misclf_info")
        # original model の repair set での間違い方をロード
        misclf_pair, tgt_label, _ = identfy_tgt_misclf(misclf_info_dir, tgt_split="repair", tgt_rank=tgt_rank, misclf_type=misclf_type)
        logger.info(f"misclf_pair: {misclf_pair}, tgt_label: {tgt_label}")
        # misclf_typeがsrc_tgtの場合は，slabel, tlabelを取得してその間違えかたをしたindexを取得
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
        metrics_dic["tgt_misclf_cnt_old"] = tgt_misclf_cnt_old
        metrics_dic["tgt_misclf_cnt_new"] = tgt_misclf_cnt_new
        metrics_dic["diff_tgt_misclf_cnt"] = tgt_misclf_cnt_new - tgt_misclf_cnt_old
        metrics_dic["new_injected_faults"] = new_injected_faults
        
    logger.info(f"metrics_dic:\n{metrics_dic}")
    # metricsをSave
    with open(metrics_dir, "w") as f:
        json.dump(metrics_dic, f, indent=4)
    logger.info(f"metrics are saved in {metrics_dir}")

    # 変更後の対象重み
    wi_new = vit_from_last_layer.base_model_last_layer.intermediate.dense.weight.data
    wo_new = vit_from_last_layer.base_model_last_layer.output.dense.weight.data
    # torchからnumpyにしてコピー
    wi_new = wi_new.cpu().numpy().copy()
    wo_new = wo_new.cpu().numpy().copy()
    # 変更された重みだけを取り出す
    wi_old_tgt = wi_old[pos_before[:, 0], pos_before[:, 1]]
    wo_old_tgt = wo_old[pos_after[:, 0], pos_after[:, 1]]
    wi_new_tgt = wi_new[pos_before[:, 0], pos_before[:, 1]]
    wo_new_tgt = wo_new[pos_after[:, 0], pos_after[:, 1]]
    # 修正履歴の重みの値のviolin plotをSave
    draw_weight_change(wi_old_tgt, wo_old_tgt, wi_new_tgt, wo_new_tgt, save_dir, setting_id)
    