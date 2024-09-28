import os, sys, time, pickle, json
import argparse
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import evaluate
import torch
from datasets import load_from_disk
from transformers import ViTForImageClassification
from utils.helper import get_device, json2dict
from utils.vit_util import transforms, transforms_c100, ViTFromLastLayer, localize_weights, localize_weights_random, src_tgt_selection, tgt_selection
from utils.data_util import make_batch_of_label
from utils.constant import ViTExperiment
from utils.log import set_exp_logging
from logging import getLogger

logger = getLogger("base_logger")

if __name__ == "__main__":
    # データセットをargparseで受け取る
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str)
    parser.add_argument('k', type=int, help="the fold id (0 to K-1)")
    parser.add_argument('tgt_rank', type=int, help="the rank of the target misclassification type")
    parser.add_argument('n', type=int, help="the factor for the number of neurons to fix")
    parser.add_argument("--fl_method", type=str, help="the method used for FL", default="vdiff")
    parser.add_argument('--misclf_type', type=str, help="the type of misclassification (src_tgt or tgt)", default="tgt")
    
    args = parser.parse_args()
    ds_name = args.ds
    k = args.k
    tgt_rank = args.tgt_rank
    n = args.n
    fl_method = args.fl_method
    misclf_type = args.misclf_type
    print(f"ds_name: {ds_name}, fold_id: {k}, tgt_rank: {tgt_rank}, n: {n}, fl_method: {fl_method}, misclf_type: {misclf_type}")

    # TODO: あとでrandomly weights selectionも実装
    if fl_method == "random":
        NotImplementedError, "randomly weights selection is not implemented yet."
    
    # pretrained modelのディレクトリ
    pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR.format(k=k)
    # 結果とかログの保存先を先に作っておく
    save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_weights_location")
    location_save_path = os.path.join(save_dir, f"location_n{n}_{fl_method}.npy")
    os.makedirs(save_dir, exist_ok=True)
    # このpythonのファイル名を取得
    this_file_name = os.path.basename(__file__).split(".")[0]
    exp_name = f"{this_file_name}_n{n}"
    # loggerの設定をして設定情報を表示
    logger = set_exp_logging(exp_dir=save_dir, exp_name=exp_name)
    logger.info(f"ds_name: {ds_name}, fold_id: {k}, tgt_rank: {tgt_rank}, n: {n}, fl_method: {fl_method}, misclf_type: {misclf_type}")

    
    tgt_split = "repair" # NOTE: we only use repair split for repairing
    tgt_layer = 11 # NOTE: we only use the last layer for repairing
    logger.info(f"tgt_layer: {tgt_layer}, tgt_split: {tgt_split}")

    # tgt_rankの誤分類情報を取り出す
    misclf_info_dir = os.path.join(pretrained_dir, "misclf_info")
    # インデックスのロード
    with open(os.path.join(misclf_info_dir, f"{tgt_split}_mis_indices.pkl"), "rb") as f:
        mis_indices = pickle.load(f)
    # ランキングのロード
    with open(os.path.join(misclf_info_dir, f"{tgt_split}_mis_ranking.pkl"), "rb") as f:
        mis_ranking = pickle.load(f)
    # metrics dictのロード
    with open(os.path.join(misclf_info_dir, f"{tgt_split}_met_dict.pkl"), "rb") as f:
        met_dict = pickle.load(f)
    if misclf_type == "src_tgt":
        slabel, tlabel, tgt_mis_indices = src_tgt_selection(mis_ranking, mis_indices, tgt_rank)
        tgt_mis_indices = mis_indices[slabel][tlabel]
        logger.info(f"tgt_misclf: {slabel} -> {tlabel}, len(tgt_mis_indices): {len(tgt_mis_indices)}")
        misclf_pair = (slabel, tlabel)
        tgt_label = None
    elif misclf_type == "tgt":
        tlabel, tgt_mis_indices = tgt_selection(met_dict, mis_indices, tgt_rank, used_met="f1")
        logger.info(f"tgt_misclf: {tlabel}, len(tgt_mis_indices): {len(tgt_mis_indices)}")
        tgt_label = tlabel
        misclf_pair = None
    else:
        NotImplementedError, f"misclf_type: {misclf_type}"

    # ===============================================
    # localization phase
    # ===============================================

    if fl_method == "vdiff":
        localizer = localize_weights
    elif fl_method == "random":
        localizer = localize_weights_random

    vscore_before_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", "vscores_before")
    vscore_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", "vscores")
    vscore_after_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", "vscores_after")
    # localizationを実行
    pos_before, pos_after = localizer(vscore_before_dir, vscore_dir, vscore_after_dir, tgt_layer, n, misclf_pair=misclf_pair, tgt_label=tgt_label)
    # log表示
    logger.info(f"pos_before={pos_before}")
    logger.info(f"pos_after={pos_after}")
    logger.info(f"num(pos_to_fix)=num(pos_before)+num(pos_before)={len(pos_before)}+{len(pos_after)}={len(pos_before)+len(pos_after)}")

    # 位置情報を保存
    np.save(location_save_path, (pos_before, pos_after))
    logger.info(f"saved location information to {location_save_path}")