import os, sys, time, pickle, json, math
from tqdm import tqdm
from collections import defaultdict
from itertools import product
import numpy as np
from utils.helper import get_device, json2dict
from utils.vit_util import localize_neurons, localize_neurons_random, localize_weights, localize_weights_random, identfy_tgt_misclf
from utils.constant import ViTExperiment, Experiment1, ExperimentRepair1, ExperimentRepair2, Experiment4
from utils.log import set_exp_logging
from logging import getLogger

logger = getLogger("base_logger")


def main(ds_name, k, tgt_rank, misclf_type, fpfn, fl_target, n):
    print(f"ds_name: {ds_name}, fold_id: {k}, tgt_rank: {tgt_rank}, misclf_type: {misclf_type}, fpfn: {fpfn}, fl_target: {fl_target}, n: {n}")
    
    # Determine the target to modify
    if fl_target == "neuron":
        fl_methods = {
            "vdiff": localize_neurons,
            "random": localize_neurons_random,
        }
    elif fl_target == "weight":
        fl_methods = {
            "vdiff": localize_weights,
            "random": localize_weights_random,
        }
    else:
        raise ValueError(f"fl_target: {fl_target} is not supported.")
    
    # Create save directories for results and logs in advance
    # Directory for pretrained model
    pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR.format(k=k)
    save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_weights_location")
    if misclf_type == "all":
        save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"all_weights_location")
    if fpfn is not None and misclf_type == "tgt":
        save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_{fpfn}_weights_location")
    os.makedirs(save_dir, exist_ok=True)
        
    # Get this Python file name
    this_file_name = os.path.basename(__file__).split(".")[0]
    exp_name = f"exp-fl-1_{this_file_name}_n{n}"
    # Set up logger and display configuration information
    logger = set_exp_logging(exp_dir=save_dir, exp_name=exp_name)
    logger.info(f"ds_name: {ds_name}, fold_id: {k}, tgt_rank: {tgt_rank}, n: {n}, misclf_type: {misclf_type}")

    # Extract misclassification information for tgt_rank
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
    # Execute localization
    for method_name, fl_method in fl_methods.items():
        print(f"fl_target: {fl_target}, method_name: {method_name}")
        location_save_path = os.path.join(save_dir, f"exp-fl-1_location_n{n}_{fl_target}_{method_name}.npy")
        st = time.perf_counter()
        ret = fl_method(vscore_before_dir, vscore_dir, vscore_after_dir, tgt_layer, n, misclf_pair=misclf_pair, tgt_label=tgt_label, fpfn=fpfn)
        et = time.perf_counter()
        logger.info(f"localization time: {et-st} sec.")
        # For neuron-level
        if fl_target == "neuron":
            places_to_fix, _ = ret
            # Display logs
            logger.info(f"places_to_fix={places_to_fix}")
            logger.info(f"num(pos_to_fix)={len(places_to_fix)}")
            print(f"num(pos_to_fix)={len(places_to_fix)}")
            # Save position information
            np.save(location_save_path, places_to_fix)
            logger.info(f"saved location information to {location_save_path}")
        # For weight-level
        elif fl_target == "weight":
            pos_before, pos_after = ret
            # Display logs
            logger.info(f"pos_before={pos_before}")
            logger.info(f"pos_after={pos_after}")
            logger.info(f"num(pos_to_fix)=num(pos_before)+num(pos_before)={len(pos_before)}+{len(pos_after)}={len(pos_before)+len(pos_after)}")
            print(f"num(pos_to_fix)=num(pos_before)+num(pos_before)={len(pos_before)}+{len(pos_after)}={len(pos_before)+len(pos_after)}")
            # Save position information
            np.save(location_save_path, (pos_before, pos_after))
            logger.info(f"saved location information to {location_save_path}")

if __name__ == "__main__":
    ds = "c100"
    # k_list = range(5)
    k_list = [0]
    tgt_rank_list = range(1, 4)
    misclf_type_list = ["all", "src_tgt", "tgt"]
    fpfn_list = [None, "fp", "fn"]
    fl_target_list = ["weight"]
    # fl_target_list = ["neuron", "weight"]
    # exp_list = [Experiment1, ExperimentRepair1, ExperimentRepair2]
    exp_list = [Experiment4]
    
    for exp in exp_list:
        for k, tgt_rank, misclf_type, fpfn, fl_target in product(k_list, tgt_rank_list, misclf_type_list, fpfn_list, fl_target_list):
            if (misclf_type == "src_tgt" or misclf_type == "all") and fpfn is not None:
                continue
            if fl_target == "neuron":
                n = exp.NUM_IDENTIFIED_NEURONS
            elif fl_target == "weight":
                n = exp.NUM_IDENTIFIED_WEIGHTS
            main(ds, k, tgt_rank, misclf_type, fpfn, fl_target, n)
        