"""
Obtain neurons with large Vdiff values for a specific misclassification type.
"""

import os, sys, time
import argparse
import pickle
from tqdm import tqdm
import numpy as np
import torch
from itertools import product
from datasets import load_from_disk
from transformers import ViTForImageClassification
from utils.helper import get_device
from utils.vit_util import transforms, transforms_c100, get_vscore, identfy_tgt_misclf, maybe_initialize_repair_weights_
from utils.constant import ViTExperiment
from utils.log import set_exp_logging
from logging import getLogger

logger = getLogger("base_logger")

def main(ds_name, k, tgt_rank, misclf_type, fpfn, abs, covavg):
    print(f"ds_name: {ds_name}, fold_id: {k}, tgt_rank: {tgt_rank}, misclf_type: {misclf_type}, fpfn: {fpfn}, abs: {abs}, covavg: {covavg}")

    # Load dataset
    dataset_dir = ViTExperiment.DATASET_DIR
    exp_obj = getattr(ViTExperiment, ds_name.replace("-", "_"))
    ds = load_from_disk(os.path.join(dataset_dir, f"{ds_name}_fold{k}"))
    pretrained_dir = exp_obj.OUTPUT_DIR.format(k=k)

    # Set different variables for each dataset
    if ds_name == "c10" or ds_name == "tiny-imagenet":
        tf_func = transforms
        label_col = "label"
    elif ds_name == "c100":
        tf_func = transforms_c100
        label_col = "fine_label"
    else:
        NotImplementedError

    tgt_pos = ViTExperiment.CLS_IDX
    prefix = ("vscore_abs" if abs else "vscore") + ("_covavg" if covavg else "")
    # Get device (cuda or cpu)
    device = get_device()
    # Load dataset (takes some time only on first load)
    tgt_split_names = list(ds.keys()) # [train, repair, test]
    # Only use repair split for target
    target_tgt_split_names = ["repair"]
    # Ensure all target_tgt_split_names are included in tgt_split_names
    assert all([tgt_split_name in tgt_split_names for tgt_split_name in target_tgt_split_names]), f"target_tgt_split_names should be subset of tgt_split_names"
    # Get labels
    labels = {
        "train": np.array(ds["train"][label_col]),
        "repair": np.array(ds["repair"][label_col]),
        "test": np.array(ds["test"][label_col])
    }
    # Apply preprocessing in real-time when loaded
    ds_preprocessed = ds.with_transform(tf_func)
    # Load pretrained model
    this_file_name = os.path.basename(__file__).split(".")[0]
    logger = set_exp_logging(exp_dir=pretrained_dir, exp_name=this_file_name)
    logger.info(f"ds_name: {ds_name}, fold_id: {k}, tgt_rank: {tgt_rank}, misclf_type: {misclf_type}, fpfn: {fpfn}")
    # Load pretrained model
    model, loading_info = ViTForImageClassification.from_pretrained(pretrained_dir, output_loading_info=True)
    model.to(device).eval()
    model = maybe_initialize_repair_weights_(model, loading_info["missing_keys"])
    end_li = model.vit.config.num_hidden_layers
    batch_size = ViTExperiment.BATCH_SIZE
    
    # Extract misclassification information for tgt_rank
    tgt_split = "repair"
    misclf_info_dir = os.path.join(pretrained_dir, "misclf_info")
    misclf_pair, tgt_label, tgt_mis_indices = identfy_tgt_misclf(misclf_info_dir, tgt_split=tgt_split, tgt_rank=tgt_rank, misclf_type=misclf_type, fpfn=fpfn)
    logger.info(f"misclf_pair: {misclf_pair}, tgt_label: {tgt_label}, len(tgt_mis_indices): {len(tgt_mis_indices)}")
    if misclf_type == "src_tgt":
        slabel, tlabel = misclf_pair
    elif misclf_type == "tgt":
        tlabel = tgt_label

    logger.info(f"Process {tgt_split}...")
    t1 = time.perf_counter()
    # Create necessary directories beforehand
    vscore_med_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", "vscores")
    os.makedirs(vscore_med_dir, exist_ok=True)
    # Assumption: v-scores for the entire dataset have already been computed
    # So, for correctly classified samples, reuse existing v-scores
    for vname in ["vscores_before", "vscores_after", "vscores"]:
        if vname == "vscores_before" or vname == "vscores_after":
            continue
        cor_vscore_path = os.path.join(pretrained_dir, vname, f"{prefix}_l1tol{end_li}_all_label_ori_{tgt_split}_cor.npy")
        tgt_vscore_path = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", vname, f"{prefix}_l1tol{end_li}_all_label_ori_{tgt_split}_cor.npy")
        os.system(f"cp {cor_vscore_path} {tgt_vscore_path}")
    all_bhs = []
    all_ahs = []
    all_mhs = []
    all_logits = []
    # Loop over dataset batch (only for misclassified samples)
    tgt_ds = ds_preprocessed[tgt_split].select(indices=tgt_mis_indices)
    for entry_dic in tqdm(tgt_ds.iter(batch_size=batch_size), total=len(tgt_ds)//batch_size+1):
        x, y = entry_dic["pixel_values"].to(device), entry_dic["labels"]
        output = model.forward(x, tgt_pos=tgt_pos, output_hidden_states_before_layernorm=False, output_intermediate_states=True)
        # Get intermediate states corresponding to CLS token
        num_layer = len(output.intermediate_states)
        bhs, ahs, mhs = [], [], []
        for i in range(num_layer):
            # output.intermediate_states[i] is a tuple of (before neurons, mid neurons, after neurons)
            bhs.append(output.intermediate_states[i][0])
            mhs.append(output.intermediate_states[i][1])
            ahs.append(output.intermediate_states[i][2]) # dimensions are layer-first here
        bhs = np.array(bhs).transpose(1, 0, 2) # (batch_size, num_layers, num_neurons)
        mhs = np.array(mhs).transpose(1, 0, 2) # (batch_size, num_layers, num_neurons)
        ahs = np.array(ahs).transpose(1, 0, 2) # (batch_size, num_layers, num_neurons)
        all_bhs.append(bhs)
        all_mhs.append(mhs)
        all_ahs.append(ahs)
        logits = output.logits.cpu().detach().numpy()
        all_logits.append(logits)
    all_logits = np.concatenate(all_logits) # (num_samples, num_labels)
    all_pred_labels = np.argmax(all_logits, axis=-1) # (num_samples, )
    if misclf_type == "src_tgt":
        # Verify all predictions are slabel
        assert all(all_pred_labels == slabel), f"all_pred_labels: {all_pred_labels}"
    elif misclf_type == "tgt":
        for pred_l, true_l in zip(all_pred_labels, labels[tgt_split][tgt_mis_indices]):
            # Ensure prediction and true label are different
            assert pred_l != true_l, f"pred_l: {pred_l}, true_l: {true_l}"
    all_bhs = np.concatenate(all_bhs)
    all_ahs = np.concatenate(all_ahs)
    all_mhs = np.concatenate(all_mhs)
    logger.info(f"all_bhs: {all_bhs.shape}, all_ahs: {all_ahs.shape}, all_mhs: {all_mhs.shape}, all_logits: {all_logits.shape}")

    t2 = time.perf_counter()
    # Compute v-scores for correct/incorrect samples and save
    for all_states, vscore_dir in zip([all_mhs], [vscore_med_dir]): # using mid-states
        vscore_per_layer = []
        for tgt_layer in range(end_li):
            logger.info(f"tgt_layer: {tgt_layer}")
            tgt_mid_state = all_states[:, tgt_layer, :] # (num_samples, num_neurons)
            vscore = get_vscore(tgt_mid_state, abs=abs, covavg=covavg) # (num_neurons, )
            vscore_per_layer.append(vscore)
        vscores = np.array(vscore_per_layer) # (num_layers, num_neurons)
        # Save vscores
        ds_type = f"ori_{tgt_split}" if fpfn is None else f"ori_{tgt_split}_{fpfn}"
        if misclf_type == "src_tgt":
            vscore_save_path = os.path.join(vscore_dir, f"{prefix}_l1tol{end_li}_{slabel}to{tlabel}_{ds_type}_mis.npy")
        elif misclf_type == "tgt":
            vscore_save_path = os.path.join(vscore_dir, f"{prefix}_l1tol{end_li}_{tlabel}_{ds_type}_mis.npy")
        np.save(vscore_save_path, vscores)
        logger.info(f"vscore ({vscores.shape}) saved at {vscore_save_path}")
        print(f"vscore ({vscores.shape}) saved at {vscore_save_path}")
    t3 = time.perf_counter()
    t_collect_hs = t2 - t1
    t_vscore = t3 - t2
    total_time = t3 - t1
    logger.info(f"total_time: {total_time} sec, collect_hs: {t_collect_hs} sec, vscore: {t_vscore} sec")

if __name__ == "__main__":
    # Accept dataset via argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str)
    parser.add_argument('k', nargs="?", type=list, help="the fold id (0 to K-1)")
    parser.add_argument('tgt_rank', nargs="?", type=list, help="the rank of the target misclassification type")
    parser.add_argument('--misclf_type', type=str, help="the type of misclassification (src_tgt or tgt)", default="tgt")
    parser.add_argument("--fpfn", type=str, help="the type of misclassification (fp or fn)", default=None, choices=["fp", "fn"])
    parser.add_argument("--run_all", action="store_true", help="run all settings")
    args = parser.parse_args()
    ds = args.ds
    k_list = args.k
    tgt_rank_list = args.tgt_rank
    misclf_type = args.misclf_type
    fpfn = args.fpfn
    run_all = args.run_all

    if run_all:
        # Error if run_all is true but k and tgt_rank are specified
        assert k_list is None and tgt_rank_list is None, "run_all and k_list or tgt_rank_list cannot be specified at the same time"
        k_list = [0]
        tgt_rank_list = range(1, 4)
        misclf_type_list = ["src_tgt", "tgt"]
        fpfn_list = [None, "fp", "fn"]
        abs_list = [True]
        covavg_list = [False]

        for k, tgt_rank, misclf_type, fpfn, abs, covavg in product(k_list, tgt_rank_list, misclf_type_list, fpfn_list, abs_list, covavg_list):
            if misclf_type == "src_tgt" and fpfn is not None:
                continue
            main(ds, k, tgt_rank, misclf_type, fpfn, abs, covavg)
    else:
        assert k_list is not None and tgt_rank_list is not None, "k_list and tgt_rank_list should be specified"
        for k, tgt_rank in zip(k_list, tgt_rank_list):
            main(ds, k, tgt_rank, misclf_type, fpfn)