import os, sys, time, pickle, json, math
from tqdm import tqdm
from collections import defaultdict
from itertools import product
import numpy as np
import pandas as pd
import torch
from utils.helper import get_device, json2dict
from utils.vit_util import identfy_tgt_misclf, localize_neurons_with_mean_activation, get_batched_hs, get_batched_labels, ViTFromLastLayer, get_ori_model_predictions
from utils.constant import ViTExperiment, ExperimentRepair1, Experiment3, ExperimentRepair2, Experiment1, Experiment4
from utils.log import set_exp_logging
from utils.arachne import calculate_top_n_flattened, calculate_bi_fi
from logging import getLogger
from datasets import load_from_disk
from transformers import ViTForImageClassification
import torch
import torch.optim as optim

logger = getLogger("base_logger")
device = get_device()

def main(ds_name, k, tgt_rank, misclf_type, fpfn, n, beta=0.5):
    print(f"ds_name: {ds_name}, fold_id: {k}, tgt_rank: {tgt_rank}, misclf_type: {misclf_type}, fpfn: {fpfn}, n: {n}")
    
    ts = time.perf_counter()
    
    # Load dataset (because we want true_labels)
    ds_dirname = f"{ds_name}_fold{k}"
    ds = load_from_disk(os.path.join(ViTExperiment.DATASET_DIR, ds_dirname))
    label_col = "fine_label"
    # Get labels (not shuffled)
    labels = {
        "train": np.array(ds["train"][label_col]),
        "repair": np.array(ds["repair"][label_col]),
        "test": np.array(ds["test"][label_col])
    }
    tgt_pos = ViTExperiment.CLS_IDX
    
    # Create save destination for results and logs in advance
    # Pretrained model directory
    pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR.format(k=k)
    save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_weights_location")
    if misclf_type == "all":
        save_dir = os.path.join(pretrained_dir, f"all_weights_location")
    if fpfn is not None and misclf_type == "tgt":
        save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_{fpfn}_weights_location")
    os.makedirs(save_dir, exist_ok=True)

    # Extract misclassification information for tgt_rank
    tgt_split = "repair" # NOTE: we only use repair split for repairing
    tgt_layer = 11 # NOTE: we only use the last layer for repairing
    misclf_info_dir = os.path.join(pretrained_dir, "misclf_info")
    misclf_pair, tgt_label, tgt_mis_indices = identfy_tgt_misclf(misclf_info_dir, tgt_split=tgt_split, tgt_rank=tgt_rank, misclf_type=misclf_type, fpfn=fpfn)
    
    # Get correct/incorrect indices for each sample in the repair set of the original model
    pred_res_dir = os.path.join(pretrained_dir, "pred_results", "PredictionOutput")
    if misclf_type == "tgt":
        ori_pred_labels, is_correct, indices_to_correct, is_correct_others, indices_to_correct_others = get_ori_model_predictions(pred_res_dir, labels, tgt_split=tgt_split, misclf_type=misclf_type, tgt_label=tgt_label)
    else:
        ori_pred_labels, is_correct, indices_to_correct = get_ori_model_predictions(pred_res_dir, labels, tgt_split=tgt_split, misclf_type=misclf_type, tgt_label=tgt_label)
    print(f"len(indices_to_correct): {len(indices_to_correct)}, len(tgt_mis_indices): {len(tgt_mis_indices)}")
    
    # Load intermediate neuron value cache
    mid_cache_dir = os.path.join(pretrained_dir, f"cache_states_{tgt_split}")
    mid_save_path = os.path.join(mid_cache_dir, f"intermediate_states_l{tgt_layer}.pt")
    cached_mid_states = torch.load(mid_save_path, map_location="cpu") # (number of samples in tgt_split (10000), number of intermediate neurons (3072))
    # Convert cached_mid_states to numpy array
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
    # Execute localization using vscore and mean_activation
    places_to_neuron, tgt_neuron_score, neuron_scores = localize_neurons_with_mean_activation(vscore_before_dir, vscore_dir, vscore_after_dir, tgt_layer, n=None, intermediate_states=cached_mid_states, tgt_mis_indices=tgt_mis_indices, misclf_pair=misclf_pair, tgt_label=tgt_label, fpfn=fpfn, return_all_neuron_score=True, vscore_abs=True)
    # Log display
    # logger.info(f"places_to_neuron={places_to_neuron}")
    # logger.info(f"num(pos_to_fix)={len(places_to_neuron)}")
    # Save position information
    # print(f"len(places_to_neuron): {len(places_to_neuron)}")
    # print(f"tgt_neuron_score.shape: {tgt_neuron_score.shape}")
    # print(f"tgt_neuron_score: {tgt_neuron_score}")
    print(f"neuron_scores.shape: {neuron_scores.shape}")
    print(f"neuron_scores: {neuron_scores}")
    
    # ============================================================
    # At this point, we have calculated neuron-wise scores using Vdiff x Use_i, so next we will identify weights using gradients as well.
    # ============================================================
    
    # Directory for cache storage
    cache_dir = os.path.join(pretrained_dir, f"cache_hidden_states_before_layernorm_{tgt_split}")
    cache_path = os.path.join(cache_dir, f"hidden_states_before_layernorm_{tgt_layer}.npy")
    # Confirm that cache_path exists
    assert os.path.exists(cache_path), f"cache_path: {cache_path} does not exist."
    # Get batches using vit_utils functions
    batch_size = ViTExperiment.BATCH_SIZE
    
    # Split correct samples (I_pos) and incorrect samples (I_neg)
    correct_batched_hidden_states = get_batched_hs(cache_path, batch_size, indices_to_correct)
    correct_batched_labels = get_batched_labels(labels[tgt_split], batch_size, indices_to_correct)
    incorrect_batched_hidden_states = get_batched_hs(cache_path, batch_size, tgt_mis_indices)
    incorrect_batched_labels = get_batched_labels(labels[tgt_split], batch_size, tgt_mis_indices)
    
    # Check shape of hidden_states_before_layernorm
    assert len(correct_batched_hidden_states) == len(correct_batched_labels), f"len(correct_batched_hidden_states): {len(correct_batched_hidden_states)}, len(correct_batched_labels): {len(correct_batched_labels)}"
    assert len(incorrect_batched_hidden_states) == len(incorrect_batched_labels), f"len(incorrect_batched_hidden_states): {len(incorrect_batched_hidden_states)}, len(incorrect_batched_labels): {len(incorrect_batched_labels)}"
    
    # Load model necessary for obtaining loss gradients
    model = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
    model.eval()
    vit_from_last_layer = ViTFromLastLayer(model)
    vit_from_last_layer.eval()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # =========================================
    # Calculate BI and FI
    # =========================================
    
    # Integrate overall grad_loss and fwd_imp
    grad_loss_list = [] # [grad_loss for Wbef, grad_loss for Waft]
    fwd_imp_list = []  # [fwd_imp for Wbef, fwd_imp for Waft]
    # BI, FI for correct samples
    print(f"Calculating BI and FI... (correct samples)")
    pos_results = calculate_bi_fi(
        indices_to_correct,
        correct_batched_hidden_states,
        correct_batched_labels,
        vit_from_last_layer,
        optimizer,
        tgt_pos,
    )
    # BI, FI for incorrect samples
    print(f"Calculating BI and FI... (incorrect samples)")
    neg_results = calculate_bi_fi(
        tgt_mis_indices,
        incorrect_batched_hidden_states,
        incorrect_batched_labels,
        vit_from_last_layer,
        optimizer,
        tgt_pos,
    )
    # Calculate for each of "before" and "after"
    for ba in ["before", "after"]:
        # Gradient Loss (Arachne Algorithm1 L6)
        grad_loss = neg_results[ba]["bw"] / (1 + pos_results[ba]["bw"])
        print(f"{ba} - grad_loss.shape: {grad_loss.shape}")  # shape: (out_dim, in_dim)
        grad_loss_list.append(grad_loss)

        # Forward Impact (Arachne Algorithm1 L9)
        fwd_imp = neg_results[ba]["fw"] / (1 + pos_results[ba]["fw"])
        print(f"{ba} - fwd_imp.shape: {fwd_imp.shape}")  # shape: (out_dim, in_dim)
        fwd_imp_list.append(fwd_imp)

        # Get out_dim of "before"
        if ba == "before":
            out_dim_before = grad_loss.shape[0]  # out_dim_before = out_dim

    # Weighted sum of forward/backward impacts
    print("Calculating top n for target weights...")
    identified_indices = calculate_top_n_flattened(grad_loss_list, fwd_imp_list, n=None)
    print(f"len(identified_indices['bef']): {len(identified_indices['bef'])}, len(identified_indices['aft']): {len(identified_indices['aft'])}")
    
    # Store separately as "before" and "after"
    pos_before = identified_indices["bef"]
    pos_after = identified_indices["aft"]
    # Score for each weight
    weighted_scores = identified_indices["scores"]
    assert len(weighted_scores) == len(pos_before) + len(pos_after), f"len(weighted_scores): {len(weighted_scores)}, len(pos_before): {len(pos_before)}, len(pos_after): {len(pos_after)}"
    
    # Output results
    print(f"pos_before.shape: {pos_before.shape}, pos_after.shape: {pos_after.shape}")
    print(f"len(weighted scores): {len(weighted_scores)}")
    
    # From here, fusion of Vdiff and BL =====================
    # Multiply the weight-wise scores calculated by BL by the Vdiff coefficient
    wscore_gated = weighted_scores.copy()  # Copy and overwrite (in-place gating)
    
    num_wbef = len(pos_before)
    # -------------------------------
    # 1) Wbef part (rows=3072, columns=768)
    # Rows are intermediate neurons -> neuron_scores axis
    # -------------------------------
    wbef_2d = wscore_gated[:num_wbef].reshape(3072, 768)
    # neuron_scores: shape=(3072,)
    # => neuron_scores[:, np.newaxis]: shape=(3072,1)
    # => Corresponds to (3072,768) by broadcasting
    wbef_2d *= (1.0 + beta * neuron_scores[:, np.newaxis])

    # -------------------------------
    # 2) Waft part (rows=768, columns=3072)
    # Columns are intermediate neurons -> neuron_scores axis
    # -------------------------------
    waft_2d = wscore_gated[num_wbef:].reshape(768, 3072)
    # neuron_scores[np.newaxis,:]: shape=(1,3072)
    # => Corresponds to (768,3072) by broadcasting
    waft_2d *= (1.0 + beta * neuron_scores[np.newaxis, :])
    
    # print(len(wscore_gated)) # shape: (num_wbef + num_waft,)
    
    # Sort by score in descending order and get top n indices
    w_num = 8 * n * n
    top_n_indices = np.argsort(wscore_gated)[-w_num:][::-1]  # Get in descending order
    # Classify into bef and aft, restore to original shape (this is an important correction point)
    shape_bef = (3072, 768)
    shape_aft = (768, 3072)
    # Classify into bef and aft, restore to original shape
    top_n_bef = np.array([
        np.unravel_index(idx, shape_bef) for idx in top_n_indices if idx < num_wbef
    ])
    top_n_aft = np.array([
        np.unravel_index(idx - num_wbef, shape_aft) for idx in top_n_indices if idx >= num_wbef
    ])
    identified_indices = {"bef": top_n_bef, "aft": top_n_aft, "scores": wscore_gated[top_n_indices]}
    # Store separately as "before" and "after"
    pos_before = identified_indices["bef"]
    pos_after = identified_indices["aft"]
    print(f"pos_before.shape: {pos_before.shape}, pos_after.shape: {pos_after.shape}")
    print(f"len(weighted scores): {len(wscore_gated[top_n_indices])}")
    
    # Finally, save the position information of weights for each intermediate neuron to location_save_path
    if fpfn is not None and misclf_type == "tgt":
        location_save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_{fpfn}_weights_location")
    elif misclf_type == "all":
        location_save_dir = os.path.join(pretrained_dir, f"all_weights_location")
    else:
        location_save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_weights_location")
    location_save_path = os.path.join(location_save_dir, f"exp-fl-7_location_n{n}_beta{beta}_weight_ours.npy")
    np.save(location_save_path, (pos_before, pos_after))
    print(f"saved location information to {location_save_path}")
    # End time
    te = time.perf_counter()
    elapsed_time = te - ts
    return elapsed_time

if __name__ == "__main__":
    ds = "c100"
    # k_list = range(5)
    k_list = [0]
    tgt_rank_list = range(1, 4)
    # misclf_type_list = ["all", "src_tgt", "tgt"]
    misclf_type_list = ["src_tgt", "tgt"]
    fpfn_list = [None, "fp", "fn"]
    n_list = [Experiment1.NUM_IDENTIFIED_WEIGHTS, ExperimentRepair1.NUM_IDENTIFIED_WEIGHTS, ExperimentRepair2.NUM_IDENTIFIED_WEIGHTS, Experiment4.NUM_IDENTIFIED_WEIGHTS]
    n_str = "_".join([str(n) for n in n_list])
    beta_list = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    results = []
    for k, tgt_rank, misclf_type, fpfn, n, beta in product(k_list, tgt_rank_list, misclf_type_list, fpfn_list, n_list, beta_list):
        print(f"\nStart: ds={ds}, k={k}, n={n}, beta={beta}, tgt_rank={tgt_rank}, misclf_type={misclf_type}, fpfn={fpfn}\n====================================================================")
        if (misclf_type == "src_tgt" or misclf_type == "all") and fpfn is not None: # When misclf_type == "src_tgt" or "all", fpfn should only be None
            continue
        if misclf_type == "all" and tgt_rank != 1: # When misclf_type == "all", tgt_rank is irrelevant, so this loop should also be skipped
            continue
        elapsed_time = main(ds, k, tgt_rank, misclf_type, fpfn, n=n, beta=beta)
        results.append({"ds": ds, "k": k, "n": n, "beta": beta, "tgt_rank": tgt_rank, "misclf_type": misclf_type, "fpfn": fpfn, "elapsed_time": elapsed_time})
    # Convert results to CSV and save
    result_df = pd.DataFrame(results)
    result_df.to_csv(f"./exp-fl-7-1_time_n{n_str}.csv", index=False)