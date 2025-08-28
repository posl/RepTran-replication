import os, sys, time, pickle, json, math
from tqdm import tqdm
from collections import defaultdict
from itertools import product
import numpy as np
import pandas as pd
from utils.helper import get_device, json2dict
from utils.vit_util import transforms, transforms_c100, ViTFromLastLayer, identfy_tgt_misclf, get_ori_model_predictions, get_new_model_predictions, get_batched_hs, get_batched_labels, sample_from_correct_samples, sample_true_positive_indices_per_class

from utils.constant import ViTExperiment, Experiment1
from utils.log import set_exp_logging
from logging import getLogger
from datasets import load_from_disk
from transformers import ViTForImageClassification
import torch
import torch.optim as optim

NUM_IDENTIFIED_NEURONS = Experiment1.NUM_IDENTIFIED_NEURONS

# Get device (cuda, or cpu)
device = get_device()

def calculate_top_n_flattened(grad_loss_list, fwd_imp_list, n=None, weight_grad_loss=0.5, weight_fwd_imp=0.5):
    """
    Flatten BI, FI, calculate scores with weighted average, and get top n items.
    
    Args:
        grad_loss_list (list): List of BI [BI of W_bef, BI of W_aft]
        fwd_imp_list (list): List of FI [FI of W_bef, FI of W_aft]
        n (int): Number to get top n items, all items if None
        weight_grad_loss (float): Weight of grad_loss (range 0~1)
        weight_fwd_imp (float): Weight of fwd_imp (range 0~1)
    
    Returns:
        dict: Top n indices {"bef": [...], "aft": [...]} and their scores
    """
    # Convert BI, FI to single column
    flattened_grad_loss = torch.cat([x.flatten() for x in grad_loss_list])
    flattened_fwd_imp = torch.cat([x.flatten() for x in fwd_imp_list])

    # Get shapes of bef and aft
    shape_bef = grad_loss_list[0].shape
    shape_aft = grad_loss_list[1].shape

    # Record boundary index between bef and aft
    split_idx = grad_loss_list[0].numel()
    
    # Normalization
    grad_loss_min, grad_loss_max = flattened_grad_loss.min(), flattened_grad_loss.max()
    fwd_imp_min, fwd_imp_max = flattened_fwd_imp.min(), flattened_fwd_imp.max()

    normalized_grad_loss = (flattened_grad_loss - grad_loss_min) / (grad_loss_max - grad_loss_min + 1e-8)
    normalized_fwd_imp = (flattened_fwd_imp - fwd_imp_min) / (fwd_imp_max - fwd_imp_min + 1e-8)

    # Calculate weighted scores
    weighted_scores = (
        weight_grad_loss * normalized_grad_loss +
        weight_fwd_imp * normalized_fwd_imp
    ).detach().cpu().numpy()
    # If n is not specified, use all items
    if n is None:
        n = len(weighted_scores)
    
    # Sort by highest scores and calculate ranking
    sorted_indices = np.argsort(weighted_scores)[-n:][::-1]  # Descending order, i-th element is index of data with i-th largest weighted_scores
    ranks = np.zeros_like(sorted_indices) # 1-dimensional
    ranks[sorted_indices] = np.arange(1, len(weighted_scores) + 1)  # Descending order, i-th element is rank of data with i-th weighted_scores

    # Classify into bef and aft
    bef_indices, bef_ranks = [], []
    aft_indices, aft_ranks = [], []
    
    for idx, rank in enumerate(ranks): # NOTE: rank here is the rank of data at combined_diff[idx]
        if idx < split_idx:
            bef_indices.append(np.unravel_index(idx, shape_bef))
            bef_ranks.append(rank)
        else:
            adjusted_idx = idx - split_idx
            aft_indices.append(np.unravel_index(adjusted_idx, shape_aft))
            aft_ranks.append(rank)
    # Convert to nparray
    bef_indices = np.array(bef_indices)
    aft_indices = np.array(aft_indices)
    bef_ranks = np.array(bef_ranks)
    aft_ranks = np.array(aft_ranks)
    print(f"len(bef_indices): {len(bef_indices)}, len(aft_indices): {len(aft_indices)}")
    print(f"len(bef_ranks): {len(bef_ranks)}, len(aft_ranks): {len(aft_ranks)}")
    
    # Sort by ascending order of score ranking (1st place is largest, so descending order of scores)
    sorted_bef = np.argsort(bef_ranks)  # Ascending rank order
    bef_indices = bef_indices[sorted_bef]
    bef_ranks = bef_ranks[sorted_bef]

    sorted_aft = np.argsort(aft_ranks)  # Ascending rank order
    aft_indices = aft_indices[sorted_aft]
    aft_ranks = aft_ranks[sorted_aft]
    
    # Display all items within top 30 overall ranks
    print("Top 30:")
    rank_mask_bef = bef_ranks <= 30
    rank_mask_aft = aft_ranks <= 30
    print(f"rank_mask_bef: {rank_mask_bef}")
    print(f"rank_mask_aft: {rank_mask_aft}")
    print(f"bef_indices: {bef_indices[rank_mask_bef]}")
    print(f"aft_indices: {aft_indices[rank_mask_aft]}")
    print(f"bef_ranks: {bef_ranks[rank_mask_bef]}")
    print(f"aft_ranks: {aft_ranks[rank_mask_aft]}")
    exit()
    
    # Return in dictionary format
    return {
        "bef": bef_indices,
        "aft": aft_indices,
        "bef_ranks": bef_ranks,
        "aft_ranks": aft_ranks,
    }


def calculate_pareto_front_flattened(grad_loss_list, fwd_imp_list):
    """
    Flatten BI, FI and calculate Pareto front
    Args:
        grad_loss_list (list): List of BI [BI of W_bef, BI of W_aft]
        fwd_imp_list (list): List of FI [FI of W_bef, FI of W_aft]
    Returns:
        dict: Pareto front indices {"bef": [...], "aft": [...]}
    """
    # Convert BI, FI to single column
    flattened_grad_loss = torch.cat([x.flatten() for x in grad_loss_list])
    flattened_fwd_imp = torch.cat([x.flatten() for x in fwd_imp_list])
    
    # Get shapes of bef and aft
    shape_bef = grad_loss_list[0].shape
    shape_aft = grad_loss_list[1].shape

    # Record boundary index between bef and aft
    split_idx = grad_loss_list[0].numel()

    # Calculate Pareto front
    pareto_indices = approximate_pareto_front(flattened_grad_loss, flattened_fwd_imp)

    # Classify into bef and aft and restore to original shape
    pareto_bef = np.array([
        np.unravel_index(idx, shape_bef) for idx in pareto_indices if idx < split_idx
    ])
    pareto_aft = np.array([
        np.unravel_index(idx - split_idx, shape_aft) for idx in pareto_indices if idx >= split_idx
    ])

    return {"bef": pareto_bef, "aft": pareto_aft}


def approximate_pareto_front(flattened_bi_values, flattened_fi_values):
    """
    Calculate Pareto front from flattened data
    Args:
        flattened_bi_values (torch.Tensor): Flattened BI
        flattened_fi_values (torch.Tensor): Flattened FI
    Returns:
        list: List of indices included in Pareto front
    """
    # Convert BI, FI to numpy
    bi_values = flattened_bi_values.detach().cpu().numpy()
    fi_values = flattened_fi_values.detach().cpu().numpy()

    # Combine BI, FI as 2D points
    points = np.stack([bi_values, fi_values], axis=1)

    # Calculate Pareto front
    pareto_mask = np.ones(points.shape[0], dtype=bool)
    for i, point in enumerate(points):
        if pareto_mask[i]:
            pareto_mask[pareto_mask] &= ~(
                np.all(points[pareto_mask] <= point, axis=1) &
                np.any(points[pareto_mask] < point, axis=1)
            )

    pareto_indices = np.where(pareto_mask)[0]
    return pareto_indices


def calculate_bi_fi(indices, batched_hidden_states, batched_labels, vit_from_last_layer, optimizer, tgt_pos):
    """
    Calculate BI and FI for specified sample set (correct or incorrect) and separate into before/after.
    Args:
        indices (list): Set of sample indices
        batched_hidden_states (list): Cached hidden states for each batch
        batched_labels (list): True labels for each batch
        vit_from_last_layer (ViTFromLastLayer): Final layer wrapper for ViT model
        optimizer (torch.optim.Optimizer): PyTorch optimizer
        tgt_pos (int): Target position (usually CLS token)
    Returns:
        defaultdict: {"before": {"bw": grad_bw, "fw": grad_fw}, "after": {"bw": grad_bw, "fw": grad_fw}}
    """
    # results = defaultdict(lambda: {"bw": [], "fw": []})
    results = defaultdict(lambda: {"bw": None, "fw": None, "count": 0})  # For aggregation


    for cached_state, tls in tqdm(
        zip(batched_hidden_states, batched_labels),
        total=len(batched_hidden_states),
    ):
        optimizer.zero_grad()  # Initialize gradients for each sample

        # Forward pass
        logits = vit_from_last_layer(hidden_states_before_layernorm=cached_state, tgt_pos=tgt_pos)
        loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = loss_fn(logits, torch.tensor(tls).to(device))  # Average loss for samples in batch
        loss.backward(retain_graph=True)

        # Data for ForwardImpact calculation
        cached_state_aft_ln = vit_from_last_layer.base_model_last_layer.layernorm_after(cached_state)
        cached_state_aft_mid = vit_from_last_layer.base_model_last_layer.intermediate(cached_state_aft_ln)
        cached_state_aft_ln = cached_state_aft_ln[:, tgt_pos, :]
        cached_state_aft_mid = cached_state_aft_mid[:, tgt_pos, :]

        # Calculate BackwardImpact (BI) and ForwardImpact (FI)
        for cs, tgt_component, ba_layer in zip(
            [cached_state_aft_ln, cached_state_aft_mid],
            [vit_from_last_layer.base_model_last_layer.intermediate.dense, vit_from_last_layer.base_model_last_layer.output.dense],
            ["before", "after"],
        ):
            # BI: Gradient of loss
            grad_bw = tgt_component.weight.grad.cpu()
            # print(f"{ba_layer} - grad_bw.shape: {grad_bw.shape}")  # shape: (out_dim, in_dim)
            if results[ba_layer]["bw"] is None:
                results[ba_layer]["bw"] = grad_bw.detach().clone().cpu()
            else:
                results[ba_layer]["bw"] += grad_bw.detach().cpu()

            # FI: Gradient of logits × normalized neuron weights
            grad_out_weight = torch.autograd.grad(
                logits, tgt_component.weight, grad_outputs=torch.ones_like(logits), retain_graph=True
            )[0]
            tgt_weight_expanded = tgt_component.weight.unsqueeze(0)
            oi_expanded = cs.unsqueeze(1)
            
            # Calculate on GPU up to **
            impact_out_weight = tgt_weight_expanded * oi_expanded
            normalization_terms = impact_out_weight.sum(dim=2)
            normalized_impact_out_weight = impact_out_weight / (normalization_terms[:, :, None] + 1e-8)
            mean_normalized_impact_out_weight = normalized_impact_out_weight.mean(dim=0)
            grad_fw = (mean_normalized_impact_out_weight * grad_out_weight).cpu() # ** Return to CPU here
            # print(f"{ba_layer} - grad_fw.shape: {grad_fw.shape}")  # shape: (out_dim, in_dim)
            if results[ba_layer]["fw"] is None:
                results[ba_layer]["fw"] = grad_fw.detach().clone().cpu()
            else:
                results[ba_layer]["fw"] += grad_fw.detach().cpu()

            # Update count
            results[ba_layer]["count"] += 1
    
    # Calculate average for entire batch
    for ba_layer in ["before", "after"]:
        if results[ba_layer]["count"] > 0:
            results[ba_layer]["bw"] = results[ba_layer]["bw"] / results[ba_layer]["count"]
            results[ba_layer]["fw"] = results[ba_layer]["fw"] / results[ba_layer]["count"]
    return results

def main(ds_name, k, tgt_rank, misclf_type, fpfn, n, sample_from_correct=False, strategy="weighted"):
    sample_from_correct = True
    ts = time.perf_counter()
    
    # Set of variables that differ for each dataset
    tgt_split = "repair" # NOTE: we only use repair split for repairing
    tgt_layer = 11 # NOTE: we only use the last layer for repairing
    tgt_pos = ViTExperiment.CLS_IDX
    
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
    true_labels = labels[tgt_split]
    
    # Directory for saving exp-fl-5 results
    exp_dir = os.path.join("./exp-fl-5", f"{ds_name}_fold{k}")
    # Load pretrained model
    pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR.format(k=k)
    # Save destination for location information
    model = ViTForImageClassification.from_pretrained(os.path.join(pretrained_dir, "checkpoint-1250")).to(device)
    model.eval()
    vit_from_last_layer = ViTFromLastLayer(model)
    vit_from_last_layer.eval()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Extract misclassification information for tgt_rank
    misclf_info_dir = os.path.join(exp_dir, "misclf_info")
    _, tgt_label, tgt_mis_indices = identfy_tgt_misclf(misclf_info_dir, tgt_split=tgt_split, tgt_rank=tgt_rank, misclf_type=misclf_type, fpfn=fpfn)
    indices_to_incorrect = tgt_mis_indices
    
    # Get correct/incorrect indices for each sample in repair set of original model
    pred_res_dir = os.path.join(exp_dir, "PredictionOutput")
    if misclf_type == "tgt":
        ori_pred_labels, is_correct, indices_to_correct, is_correct_others, indices_to_correct_others = get_ori_model_predictions(pred_res_dir, labels, tgt_split=tgt_split, misclf_type=misclf_type, tgt_label=tgt_label)
    else:
        ori_pred_labels, is_correct, indices_to_correct = get_ori_model_predictions(pred_res_dir, labels, tgt_split=tgt_split, misclf_type=misclf_type, tgt_label=tgt_label)
    print(f"len(indices_to_correct): {len(indices_to_correct)}, len(indices_to_incorrect): {len(indices_to_incorrect)}")
    
    # Randomly extract a certain number from correct data for repair
    if sample_from_correct:
        # When sampling
        sampled_indices_to_correct = sample_from_correct_samples(len(indices_to_incorrect), indices_to_correct)
    else:
        # When not sampling
        sampled_indices_to_correct = indices_to_correct
    # Combine extracted correct data and all incorrect data into one dataset
    tgt_indices = sampled_indices_to_correct.tolist() + indices_to_incorrect.tolist() # .tolist() is non-destructive method
    # Ensure all tgt_indices are unique values
    assert len(tgt_indices) == len(set(tgt_indices)), f"len(tgt_indices): {len(tgt_indices)}, len(set(tgt_indices)): {len(set(tgt_indices))}"
    print(f"len(tgt_indices): {len(tgt_indices)})")
    # Extract data labels corresponding to tgt_indices
    tgt_labels = labels[tgt_split][tgt_indices]
    # Display prediction labels and true labels for each sample used in FL
    print(f"ori_pred_labels[tgt_indices]: {ori_pred_labels[tgt_indices]} (len: {len(ori_pred_labels[tgt_indices])})")
    print(f"ori_tgt_labels[tgt_indices]: {tgt_labels} (len: {len(tgt_labels)})")
    
    # Directory for saving cache
    cache_dir = os.path.join(exp_dir, f"cache_hidden_states_before_layernorm_{tgt_split}")
    cache_path = os.path.join(cache_dir, f"hidden_states_before_layernorm_{tgt_layer}.npy")
    # Confirm existence in cache_path
    assert os.path.exists(cache_path), f"cache_path: {cache_path} does not exist."
    # Get batches using vit_utils functions
    batch_size = ViTExperiment.BATCH_SIZE
    
    # Split into correct samples (I_pos) and incorrect samples (I_neg)
    correct_batched_hidden_states = get_batched_hs(cache_path, batch_size, sampled_indices_to_correct)
    correct_batched_labels = get_batched_labels(labels[tgt_split], batch_size, sampled_indices_to_correct)
    incorrect_batched_hidden_states = get_batched_hs(cache_path, batch_size, indices_to_incorrect)
    incorrect_batched_labels = get_batched_labels(labels[tgt_split], batch_size, indices_to_incorrect)
    
    assert len(correct_batched_hidden_states) == len(correct_batched_labels), f"len(correct_batched_hidden_states): {len(correct_batched_hidden_states)}, len(correct_batched_labels): {len(correct_batched_labels)}"
    
    # =========================================
    # Calculate BI and FI
    # =========================================
    
    # Integrate overall grad_loss and fwd_imp
    grad_loss_list = [] # [grad_loss for Wbef, grad_loss for Waft]
    fwd_imp_list = []  # [fwd_imp for Wbef, fwd_imp for Waft]
    # BI, FI for correct samples
    print(f"Calculating BI and FI... (correct samples)")
    pos_results = calculate_bi_fi(
        sampled_indices_to_correct,
        correct_batched_hidden_states,
        correct_batched_labels,
        vit_from_last_layer,
        optimizer,
        tgt_pos,
    )
    # BI, FI for incorrect samples
    print(f"Calculating BI and FI... (incorrect samples)")
    neg_results = calculate_bi_fi(
        indices_to_incorrect,
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

        # Get out_dim for "before"
        if ba == "before":
            out_dim_before = grad_loss.shape[0]  # out_dim_before = out_dim

    # Calculate Pareto front
    if strategy == "pareto":
        print(f"Calculating Pareto front for target weights...")
        identified_indices = calculate_pareto_front_flattened(grad_loss_list, fwd_imp_list)
    elif strategy == "weighted":
        print("Calculating top n for target weights...")
        wnum = 8 * n * n if n is not None else None
        identified_indices = calculate_top_n_flattened(grad_loss_list, fwd_imp_list, n=wnum)
        # Also want to save when n is not specified
    print(f"len(identified_indices['bef']): {len(identified_indices['bef'])}, len(identified_indices['aft']): {len(identified_indices['aft'])}")
    
    location_filename = "exp-fl-5_location_nAll_weight_bl.npy" if n is None else f"exp-fl-5_location_n{n}_weight_bl.npy"
    rank_filename = "exp-fl-5_location_nAll_weight_bl_rank.npy" if n is None else f"exp-fl-5_location_n{n}_weight_bl_rank.npy"
    # Store separately as "before" and "after"
    pos_before = identified_indices["bef"]
    pos_after = identified_indices["aft"]
    rank_before = identified_indices["bef_ranks"]
    rank_after = identified_indices["aft_ranks"]
    
    # Output results
    print(f"pos_before: {pos_before}")
    print(f"pos_after: {pos_after}")
    
    # Finally, save position information of weights for each intermediate neuron to location_save_path
    if fpfn is not None and misclf_type == "tgt":
        location_save_dir = os.path.join(exp_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_{fpfn}_weights_location")
    elif misclf_type == "all":
        location_save_dir = os.path.join(exp_dir, f"all_weights_location")
    else:
        location_save_dir = os.path.join(exp_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_weights_location")
    if not os.path.exists(location_save_dir):
        os.makedirs(location_save_dir)
    # old_location_save_path = os.path.join(location_save_dir, f"exp-fl-2_location_weight_bl.npy")
    location_save_path = os.path.join(location_save_dir, location_filename)
    rank_save_path = os.path.join(location_save_dir, rank_filename)
    np.save(location_save_path, (pos_before, pos_after))
    print(f"saved location information to {location_save_path}")
    np.save(rank_save_path, (rank_before, rank_after))
    print(f"saved rank information to {rank_save_path}")
    # End time
    te = time.perf_counter()
    elapsed_time = te - ts
    return elapsed_time
    
if __name__ == "__main__":
    ds = "c100"
    k_list = range(5)
    tgt_rank_list = range(1, 4)
    misclf_type_list = ["all", "src_tgt", "tgt"]
    fpfn_list = [None, "fp", "fn"]
    results = []
    n_list = [None]
    for k, tgt_rank, misclf_type, fpfn, n in product(k_list, tgt_rank_list, misclf_type_list, fpfn_list, n_list):
        print(f"Start: ds={ds}, k={k}, tgt_rank={tgt_rank}, misclf_type={misclf_type}, fpfn={fpfn}")
        if (misclf_type == "src_tgt" or misclf_type == "all") and fpfn is not None: # When misclf_type == "src_tgt" or "all", fpfn should only be None
            continue
        if misclf_type == "all" and tgt_rank != 1: # When misclf_type == "all", tgt_rank is irrelevant, so this loop should also be skipped
            continue
        elapsed_time = main(ds, k, tgt_rank, misclf_type, fpfn, n=n)
        results.append({"ds": ds, "k": k, "tgt_rank": tgt_rank, "misclf_type": misclf_type, "fpfn": fpfn, "elapsed_time": elapsed_time})
    # results を csv にしてSave
    result_df = pd.DataFrame(results)
    result_df.to_csv("./exp-fl-5-5_time.csv", index=False)