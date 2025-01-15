import os, sys, time, pickle, json, math
from tqdm import tqdm
from collections import defaultdict
from itertools import product
import numpy as np
import torch
from utils.helper import get_device, json2dict
from utils.vit_util import identfy_tgt_misclf, localize_neurons_with_mean_activation
from utils.constant import ViTExperiment, Experiment1, Experiment3
from utils.log import set_exp_logging
from logging import getLogger

logger = getLogger("base_logger")
device = get_device()

def main(ds_name, k, tgt_rank, misclf_type, fpfn):
    print(f"ds_name: {ds_name}, fold_id: {k}, tgt_rank: {tgt_rank}, misclf_type: {misclf_type}, fpfn: {fpfn}")
    
    # 変更する対象を決定
    n_ratio = Experiment3.NUM_IDENTIFIED_NEURONS_RATIO # exp-fl-3.md参照
    w_num = Experiment3.NUM_TOTAL_WEIGHTS
    
    # 結果とかログの保存先を先に作っておく
    # pretrained modelのディレクトリ
    pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR.format(k=k)
    save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_weights_location")
    if misclf_type == "all":
        save_dir = os.path.join(pretrained_dir, f"all_weights_location")
    if fpfn is not None and misclf_type == "tgt":
        save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_{fpfn}_weights_location")
    os.makedirs(save_dir, exist_ok=True)
        
    # このpythonのファイル名を取得
    this_file_name = os.path.basename(__file__).split(".")[0]
    exp_name = f"exp-fl-3_{this_file_name}_n{n_ratio}"
    # loggerの設定をして設定情報を表示
    logger = set_exp_logging(exp_dir=save_dir, exp_name=exp_name)
    logger.info(f"ds_name: {ds_name}, fold_id: {k}, tgt_rank: {tgt_rank}, n_ratio: {n_ratio}, w_num: {w_num}, misclf_type: {misclf_type}")

    # tgt_rankの誤分類情報を取り出す
    tgt_split = "repair" # NOTE: we only use repair split for repairing
    tgt_layer = 11 # NOTE: we only use the last layer for repairing
    logger.info(f"tgt_layer: {tgt_layer}, tgt_split: {tgt_split}")
    misclf_info_dir = os.path.join(pretrained_dir, "misclf_info")
    misclf_pair, tgt_label, tgt_mis_indices = identfy_tgt_misclf(misclf_info_dir, tgt_split=tgt_split, tgt_rank=tgt_rank, misclf_type=misclf_type, fpfn=fpfn)
    logger.info(f"misclf_pair: {misclf_pair}, tgt_label: {tgt_label}, len(tgt_mis_indices): {len(tgt_mis_indices)}")
    
    # 中間ニューロン値のキャッシュのロード
    mid_cache_dir = os.path.join(pretrained_dir, f"cache_states_{tgt_split}")
    mid_save_path = os.path.join(mid_cache_dir, f"intermediate_states_l{tgt_layer}.pt")
    cached_mid_states = torch.load(mid_save_path, map_location="cpu") # (tgt_splitのサンプル数(10000), 中間ニューロン数(3072))
    # cached_mid_statesをnumpy配列にする
    cached_mid_states = cached_mid_states.detach().numpy().copy()
    print(f"cached_mid_states.shape: {cached_mid_states.shape}")

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
    # vscoreとmean_activationを用いたlocalizationを実行
    method_name = "vscore_mean_activation"
    location_save_path = os.path.join(save_dir, f"exp-fl-3_location_n{n_ratio}_w{w_num}_neuron.npy")
    st = time.perf_counter()
    places_to_fix, tgt_neuron_score = localize_neurons_with_mean_activation(vscore_before_dir, vscore_dir, vscore_after_dir, tgt_layer, n_ratio, intermediate_states=cached_mid_states, tgt_mis_indices=tgt_mis_indices, misclf_pair=misclf_pair, tgt_label=tgt_label, fpfn=fpfn)
    et = time.perf_counter()
    logger.info(f"localization time: {et-st} sec.")
    # log表示
    logger.info(f"places_to_fix={places_to_fix}")
    logger.info(f"num(pos_to_fix)={len(places_to_fix)}")
    # 位置情報を保存
    np.save(location_save_path, places_to_fix)
    logger.info(f"saved location information to {location_save_path}")
    print(f"len(places_to_fix): {len(places_to_fix)}")
    print(f"tgt_neuron_score.shape: {tgt_neuron_score.shape}")
    print(f"tgt_neuron_score: {tgt_neuron_score}")
    
    # TODO: ここまでで Vdiff x Use_i によるニューロン特定ができたので，次は勾配も使った重み特定をする．

if __name__ == "__main__":
    ds = "c100"
    k_list = range(5)
    tgt_rank_list = range(1, 6)
    misclf_type_list = ["all", "src_tgt", "tgt"]
    fpfn_list = [None, "fp", "fn"]
    for k, tgt_rank, misclf_type, fpfn in product(k_list, tgt_rank_list, misclf_type_list, fpfn_list):
        if (misclf_type == "src_tgt" or misclf_type == "all") and fpfn is not None:
            continue
        main(ds, k, tgt_rank, misclf_type, fpfn)
    