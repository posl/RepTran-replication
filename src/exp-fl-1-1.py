import os, sys, time, pickle, json, math
from tqdm import tqdm
from collections import defaultdict
from itertools import product
import numpy as np
from utils.helper import get_device, json2dict
from utils.vit_util import localize_neurons, localize_neurons_random, localize_weights, localize_weights_random, identfy_tgt_misclf
from utils.constant import ViTExperiment, Experiment1
from utils.log import set_exp_logging
from logging import getLogger

logger = getLogger("base_logger")
NUM_IDENTIFIED_NEURONS = Experiment1.NUM_IDENTIFIED_NEURONS # exp-fl-1.md参照
NUM_IDENTIFIED_WEIGHTS = Experiment1.NUM_IDENTIFIED_NEURONS # exp-fl-1.md参照

def main(ds_name, k, tgt_rank, misclf_type, fpfn, fl_target):
    print(f"ds_name: {ds_name}, fold_id: {k}, tgt_rank: {tgt_rank}, misclf_type: {misclf_type}, fpfn: {fpfn}, fl_target: {fl_target}")
    
    # 変更する対象を決定
    if fl_target == "neuron":
        n = NUM_IDENTIFIED_NEURONS
        fl_methods = {
            "vdiff": localize_neurons,
            "random": localize_neurons_random,
        }
    elif fl_target == "weight":
        n = NUM_IDENTIFIED_WEIGHTS
        fl_methods = {
            "vdiff": localize_weights,
            "random": localize_weights_random,
        }
    else:
        raise ValueError(f"fl_target: {fl_target} is not supported.")
    
    # 結果とかログの保存先を先に作っておく
    # pretrained modelのディレクトリ
    pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR.format(k=k)
    save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_weights_location")
    if misclf_type == "all":
        save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"all_weights_location")
    if fpfn is not None and misclf_type == "tgt":
        save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_{fpfn}_weights_location")
    os.makedirs(save_dir, exist_ok=True)
        
    # このpythonのファイル名を取得
    this_file_name = os.path.basename(__file__).split(".")[0]
    exp_name = f"exp-fl-1_{this_file_name}_n{n}"
    # loggerの設定をして設定情報を表示
    logger = set_exp_logging(exp_dir=save_dir, exp_name=exp_name)
    logger.info(f"ds_name: {ds_name}, fold_id: {k}, tgt_rank: {tgt_rank}, n: {n}, misclf_type: {misclf_type}")

    # tgt_rankの誤分類情報を取り出す
    tgt_split = "repair" # NOTE: we only use repair split for repairing
    tgt_layer = 11 # NOTE: we only use the last layer for repairing
    logger.info(f"tgt_layer: {tgt_layer}, tgt_split: {tgt_split}")
    misclf_info_dir = os.path.join(pretrained_dir, "misclf_info")
    misclf_pair, tgt_label, tgt_mis_indices = identfy_tgt_misclf(misclf_info_dir, tgt_split=tgt_split, tgt_rank=tgt_rank, misclf_type=misclf_type, fpfn=fpfn)
    logger.info(f"misclf_pair: {misclf_pair}, tgt_label: {tgt_label}, len(tgt_mis_indices): {len(tgt_mis_indices)}")

    # ===============================================
    # localization phase
    # ===============================================

    if misclf_type == "src_tgt" or misclf_type == "tgt":
        vscore_before_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", "vscores_before")
        vscore_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", "vscores")
        vscore_after_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", "vscores_after")
    elif misclf_type == "all":
        vscore_before_dir = os.path.join(pretrained_dir, "vscores_before")
        vscore_dir = os.path.join(pretrained_dir, "vscores")
        vscore_after_dir = os.path.join(pretrained_dir, "vscores_after")
    logger.info(f"vscore_before_dir: {vscore_before_dir}")
    logger.info(f"vscore_dir: {vscore_dir}")
    logger.info(f"vscore_after_dir: {vscore_after_dir}")
    # localizationを実行
    for method_name, fl_method in fl_methods.items():
        print(f"fl_target: {fl_target}, method_name: {method_name}")
        location_save_path = os.path.join(save_dir, f"exp-fl-1_location_n{n}_{fl_target}_{method_name}.npy")
        st = time.perf_counter()
        ret = fl_method(vscore_before_dir, vscore_dir, vscore_after_dir, tgt_layer, n, misclf_pair=misclf_pair, tgt_label=tgt_label, fpfn=fpfn)
        et = time.perf_counter()
        logger.info(f"localization time: {et-st} sec.")
        # ニューロン単位の場合
        if fl_target == "neuron":
            places_to_fix, _ = ret
            # log表示
            logger.info(f"places_to_fix={places_to_fix}")
            logger.info(f"num(pos_to_fix)={len(places_to_fix)}")
            # 位置情報を保存
            np.save(location_save_path, places_to_fix)
            logger.info(f"saved location information to {location_save_path}")
        # 重み単位の場合
        elif fl_target == "weight":
            pos_before, pos_after = ret
            # log表示
            logger.info(f"pos_before={pos_before}")
            logger.info(f"pos_after={pos_after}")
            logger.info(f"num(pos_to_fix)=num(pos_before)+num(pos_before)={len(pos_before)}+{len(pos_after)}={len(pos_before)+len(pos_after)}")
            # 位置情報を保存
            np.save(location_save_path, (pos_before, pos_after))
            logger.info(f"saved location information to {location_save_path}")

if __name__ == "__main__":
    ds = "c100"
    k_list = range(5)
    tgt_rank_list = range(1, 6)
    misclf_type_list = ["all", "src_tgt", "tgt"]
    fpfn_list = [None, "fp", "fn"]
    fl_target_list = ["neuron", "weight"]
    for k, tgt_rank, misclf_type, fpfn, fl_target in product(k_list, tgt_rank_list, misclf_type_list, fpfn_list, fl_target_list):
        if (misclf_type == "src_tgt" or misclf_type == "all") and fpfn is not None:
            continue
        main(ds, k, tgt_rank, misclf_type, fpfn, fl_target)
    