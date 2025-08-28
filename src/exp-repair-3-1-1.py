import os, sys, time, pickle, json, math
from tqdm import tqdm
from collections import defaultdict
from itertools import product
import numpy as np
import pandas as pd
from utils.helper import get_device, json2dict
from utils.vit_util import transforms, transforms_c100, ViTFromLastLayer, identfy_tgt_misclf, get_ori_model_predictions, get_new_model_predictions, get_batched_hs, get_batched_labels, sample_from_correct_samples, sample_true_positive_indices_per_class, maybe_initialize_repair_weights_

from utils.constant import ViTExperiment, Experiment1, ExperimentRepair1, ExperimentRepair2, Experiment4
from utils.log import set_exp_logging
from logging import getLogger
from datasets import load_from_disk
from transformers import ViTForImageClassification
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# Get device (cuda, or cpu)
device = get_device()

def calculate_top_n_flattened(grad_loss_list, fwd_imp_list, n, weight_grad_loss=0.5, weight_fwd_imp=0.5):
    """
    Flatten BI, FI, calculate scores with weighted average, and get top n items.
    
    Args:
        grad_loss_list (list): List of BI [BI of W_bef, BI of W_aft]
        fwd_imp_list (list): List of FI [FI of W_bef, FI of W_aft]
        n (int): Number to get top n items
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
    
    # Normalize
    grad_loss_min, grad_loss_max = flattened_grad_loss.min(), flattened_grad_loss.max()
    fwd_imp_min, fwd_imp_max = flattened_fwd_imp.min(), flattened_fwd_imp.max()

    normalized_grad_loss = (flattened_grad_loss - grad_loss_min) / (grad_loss_max - grad_loss_min + 1e-8)
    normalized_fwd_imp = (flattened_fwd_imp - fwd_imp_min) / (fwd_imp_max - fwd_imp_min + 1e-8)

    # Weighted score calculation
    weighted_scores = (
        weight_grad_loss * normalized_grad_loss +
        weight_fwd_imp * normalized_fwd_imp
    ).detach().cpu().numpy()

    # Get top n indices sorted in descending order
    top_n_indices = np.argsort(weighted_scores)[-n:][::-1]

    # Split into bef and aft, and reshape to original dimensions
    top_n_bef = np.array([
        np.unravel_index(idx, shape_bef) for idx in top_n_indices if idx < split_idx
    ])
    top_n_aft = np.array([
        np.unravel_index(idx - split_idx, shape_aft) for idx in top_n_indices if idx >= split_idx
    ])

    return {"bef": top_n_bef, "aft": top_n_aft, "scores": weighted_scores[top_n_indices]}


def calculate_pareto_front_flattened(grad_loss_list, fwd_imp_list):
    """
    Flatten BI, FI and compute Pareto front.
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

    # Compute Pareto front
    pareto_indices = approximate_pareto_front(flattened_grad_loss, flattened_fwd_imp)

    # Split into bef and aft, and reshape to original dimensions
    pareto_bef = np.array([
        np.unravel_index(idx, shape_bef) for idx in pareto_indices if idx < split_idx
    ])
    pareto_aft = np.array([
        np.unravel_index(idx - split_idx, shape_aft) for idx in pareto_indices if idx >= split_idx
    ])

    return {"bef": pareto_bef, "aft": pareto_aft}


def approximate_pareto_front(flattened_bi_values, flattened_fi_values):
    """
    Compute Pareto front from flattened BI/FI.
    Args:
        flattened_bi_values (torch.Tensor): Flattened BI
        flattened_fi_values (torch.Tensor): Flattened FI
    Returns:
        list: List of indices belonging to the Pareto front
    """
    bi_values = flattened_bi_values.detach().cpu().numpy()
    fi_values = flattened_fi_values.detach().cpu().numpy()

    points = np.stack([bi_values, fi_values], axis=1)

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
    Compute BI and FI for the specified set of samples (correct or incorrect), split into before/after.
    Args:
        indices (list): Sample indices
        batched_hidden_states (list): Cached hidden states per batch
        batched_labels (list): Labels per batch
        vit_from_last_layer (ViTFromLastLayer): ViT wrapper of the last layer
        optimizer (torch.optim.Optimizer): PyTorch optimizer
        tgt_pos (int): Target position (typically CLS token)
    Returns:
        defaultdict: {"before": {"bw": grad_bw, "fw": grad_fw}, "after": {"bw": grad_bw, "fw": grad_fw}}
    """
    results = defaultdict(lambda: {"bw": None, "fw": None, "count": 0})  # For aggregation

    for cached_state, tls in tqdm(
        zip(batched_hidden_states, batched_labels),
        total=len(batched_hidden_states),
    ):
        optimizer.zero_grad()  # Reset gradients for each sample

        # Forward pass
        logits = vit_from_last_layer(hidden_states_before_layernorm=cached_state, tgt_pos=tgt_pos)
        loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = loss_fn(logits, torch.tensor(tls).to(device))
        loss.backward(retain_graph=True)

        # ForwardImpact data
        cached_state_aft_ln = vit_from_last_layer.base_model_last_layer.layernorm_after(cached_state)
        cached_state_aft_mid = vit_from_last_layer.base_model_last_layer.intermediate(cached_state_aft_ln)
        cached_state_aft_ln = cached_state_aft_ln[:, tgt_pos, :]
        cached_state_aft_mid = cached_state_aft_mid[:, tgt_pos, :]

        # Compute BackwardImpact (BI) and ForwardImpact (FI)
        for cs, tgt_component, ba_layer in zip(
            [cached_state_aft_ln, cached_state_aft_mid],
            [vit_from_last_layer.base_model_last_layer.intermediate.dense, vit_from_last_layer.base_model_last_layer.output.dense],
            ["before", "after"],
        ):
            # BI: gradient of loss
            grad_bw = tgt_component.weight.grad.cpu()
            if results[ba_layer]["bw"] is None:
                results[ba_layer]["bw"] = grad_bw.detach().clone().cpu()
            else:
                results[ba_layer]["bw"] += grad_bw.detach().cpu()

            # FI: gradient × normalized neuron weights
            grad_out_weight = torch.autograd.grad(
                logits, tgt_component.weight, grad_outputs=torch.ones_like(logits), retain_graph=True
            )[0]
            tgt_weight_expanded = tgt_component.weight.unsqueeze(0)
            oi_expanded = cs.unsqueeze(1)
            
            impact_out_weight = tgt_weight_expanded * oi_expanded
            normalization_terms = impact_out_weight.sum(dim=2)
            normalized_impact_out_weight = impact_out_weight / (normalization_terms[:, :, None] + 1e-8)
            mean_normalized_impact_out_weight = normalized_impact_out_weight.mean(dim=0)
            grad_fw = (mean_normalized_impact_out_weight * grad_out_weight).cpu()
            if results[ba_layer]["fw"] is None:
                results[ba_layer]["fw"] = grad_fw.detach().clone().cpu()
            else:
                results[ba_layer]["fw"] += grad_fw.detach().cpu()

            results[ba_layer]["count"] += 1
    
    # Compute averages
    for ba_layer in ["before", "after"]:
        if results[ba_layer]["count"] > 0:
            results[ba_layer]["bw"] = results[ba_layer]["bw"] / results[ba_layer]["count"]
            results[ba_layer]["fw"] = results[ba_layer]["fw"] / results[ba_layer]["count"]
    return results