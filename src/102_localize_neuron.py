import os, sys, time, pickle, json
import argparse
from tqdm import tqdm
from collections import defaultdict
from itertools import product
import numpy as np
from utils.helper import get_device, json2dict
from utils.vit_util import localize_neurons, localize_neurons_random, identfy_tgt_misclf
from utils.constant import ViTExperiment
from utils.log import set_exp_logging
from logging import getLogger

logger = getLogger("base_logger")


def main(ds_name, k, tgt_rank, theta, fl_method, misclf_type, fpfn, run_all=False):
    print(
        f"ds_name: {ds_name}, fold_id: {k}, tgt_rank: {tgt_rank}, theta: {theta}, fl_method: {fl_method}, misclf_type: {misclf_type}, fpfn: {fpfn}"
    )

    # Directory for pretrained model
    pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR.format(k=k)
    # Create save directories for results and logs in advance
    save_dir = os.path.join(
        pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_neurons_location"
    )
    if misclf_type == "all":
        save_dir = os.path.join(pretrained_dir, f"all_neurons_location")
    if fpfn is not None and misclf_type == "tgt":
        save_dir = os.path.join(
            pretrained_dir,
            f"misclf_top{tgt_rank}",
            f"{misclf_type}_{fpfn}_neurons_location",
        )
    location_save_path = os.path.join(
        save_dir, f"location_theta{theta}_{fl_method}.npy"
    )
    os.makedirs(save_dir, exist_ok=True)
    proba_save_dir = os.path.join(save_dir, f"proba_theta{theta}_{fl_method}")
    os.makedirs(proba_save_dir, exist_ok=True)
    # Get this Python file name
    this_file_name = os.path.basename(__file__).split(".")[0]
    exp_name = (
        f"{this_file_name}_theta{theta}" if not run_all else f"{this_file_name}_run_all"
    )
    # Set up logger and display configuration information
    logger = set_exp_logging(exp_dir=save_dir, exp_name=exp_name)
    logger.info(
        f"ds_name: {ds_name}, fold_id: {k}, tgt_rank: {tgt_rank}, theta: {theta}, fl_method: {fl_method}, misclf_type: {misclf_type}"
    )

    # Extract misclassification information for tgt_rank
    tgt_split = "repair"  # NOTE: we only use repair split for repairing
    tgt_layer = 11  # NOTE: we only use the last layer for repairing
    logger.info(f"tgt_layer: {tgt_layer}, tgt_split: {tgt_split}")
    misclf_info_dir = os.path.join(pretrained_dir, "misclf_info")
    misclf_pair, tgt_label, tgt_mis_indices = identfy_tgt_misclf(
        misclf_info_dir,
        tgt_split=tgt_split,
        tgt_rank=tgt_rank,
        misclf_type=misclf_type,
        fpfn=fpfn,
    )
    logger.info(
        f"misclf_pair: {misclf_pair}, tgt_label: {tgt_label}, len(tgt_mis_indices): {len(tgt_mis_indices)}"
    )

    # ===============================================
    # localization phase
    # ===============================================

    if fl_method == "vdiff":
        localizer = localize_neurons
    elif fl_method == "random":
        localizer = localize_neurons_random

    if misclf_type == "src_tgt" or misclf_type == "tgt":
        vscore_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", "vscores")
    elif misclf_type == "all":
        vscore_dir = os.path.join(pretrained_dir, "vscores")
    logger.info(f"vscore_dir: {vscore_dir}")
    # Execute localization
    st = time.perf_counter()
    vmap_dic = defaultdict(np.array)
    for cor_mis in ["cor", "mis"]:
        ds_type = f"ori_{tgt_split}"
        vscore_save_path = os.path.join(
            vscore_dir, f"vscore_l1tol12_all_label_{ds_type}_{cor_mis}.npy"
        )
        if misclf_pair is not None and cor_mis == "mis":
            # misclf_pairが指定されている場合は，その対象のデータのみを取得
            assert len(misclf_pair) == 2, f"Error: {misclf_pair}"
            slabel, tlabel = misclf_pair
            vscore_save_path = os.path.join(
                vscore_dir, f"vscore_l1tol12_{slabel}to{tlabel}_{ds_type}_{cor_mis}.npy"
            )
        if tgt_label is not None and cor_mis == "mis":
            # tgt_labelが指定されている場合は，その対象のデータのみを取得
            vscore_save_path = os.path.join(
                vscore_dir, f"vscore_l1tol12_{tgt_label}_{ds_type}_{cor_mis}.npy"
            )
            if fpfn is not None:
                vscore_save_path = os.path.join(
                    vscore_dir,
                    f"vscore_l1tol12_{tgt_label}_{ds_type}_{fpfn}_{cor_mis}.npy",
                )
        vscores = np.load(vscore_save_path)
        vmap_dic[cor_mis] = vscores.T
        print(f"vscores shape ({cor_mis}): {vmap_dic[cor_mis].shape}")
    places_to_fix, tgt_vdiff = localizer(vmap_dic, tgt_layer, theta=theta)
    et = time.perf_counter()
    logger.info(f"localization time: {et-st} sec.")
    # Display logs
    logger.info(f"places_to_fix={places_to_fix}")
    logger.info(f"num(location)={len(places_to_fix)}")

    # 位置情報をSave
    np.save(location_save_path, places_to_fix)
    logger.info(f"saved location information to {location_save_path}")


if __name__ == "__main__":
    # Accept dataset via argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str)
    parser.add_argument("k", nargs="?", type=list, help="the fold id (0 to K-1)")
    parser.add_argument(
        "tgt_rank",
        nargs="?",
        type=list,
        help="the rank of the target misclassification type",
    )
    parser.add_argument(
        "theta",
        nargs="?",
        type=int,
        help="the percentage for the number of neurons to fix (%)",
    )
    parser.add_argument(
        "--misclf_type",
        type=str,
        help="the type of misclassification (src_tgt or tgt)",
        default="tgt",
    )
    parser.add_argument(
        "--fpfn",
        type=str,
        help="the type of misclassification (fp or fn)",
        default=None,
        choices=["fp", "fn"],
    )
    parser.add_argument(
        "--fl_method", type=str, help="the method used for FL", default="vdiff"
    )
    parser.add_argument("--run_all", action="store_true", help="run all settings")
    args = parser.parse_args()
    ds = args.ds
    k_list = args.k
    tgt_rank_list = args.tgt_rank
    theta_list = args.theta
    misclf_type = args.misclf_type
    fpfn = args.fpfn
    fl_method = args.fl_method
    run_all = args.run_all

    assert (
        fl_method == "vdiff" or fl_method == "random"
    ), "fl_method should be vdiff or random."

    if run_all:
        # run_allがtrueなのにkとtgt_rankが指定されている場合はエラー
        assert (
            k_list is None and tgt_rank_list is None and theta_list is None
        ), "run_all and k_list or tgt_rank_list or theta_list cannot be specified at the same time"
        k_list = range(5)
        tgt_rank_list = range(1, 4)
        theta_list = [5, 10, 15, 20, 25, 30]
        misclf_type_list = ["all", "src_tgt", "tgt"]
        fpfn_list = [None, "fp", "fn"]
        for k, tgt_rank, theta, misclf_type, fpfn in product(
            k_list, tgt_rank_list, theta_list, misclf_type_list, fpfn_list
        ):
            if (misclf_type == "src_tgt" or misclf_type == "all") and fpfn is not None:
                continue
            main(ds, k, tgt_rank, theta, fl_method, misclf_type, fpfn, run_all=run_all)
    else:
        assert (
            k_list is not None and tgt_rank_list is not None and theta_list is not None
        ), "k_list and tgt_rank_list and n_list should be specified"
        for k, tgt_rank, theta in zip(k_list, tgt_rank_list, theta_list):
            main(ds, k, tgt_rank, theta, fl_method, misclf_type, fpfn)
