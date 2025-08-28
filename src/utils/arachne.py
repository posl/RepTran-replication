import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm
from utils.helper import get_device
device = get_device()

def calculate_top_n_flattened(grad_loss_list, fwd_imp_list, n=None, weight_grad_loss=0.5, weight_fwd_imp=0.5):
    """
    Flatten BI, FI, calculate scores with weighted average, and get top n items.
    
    Args:
        grad_loss_list (list): List of BI [BI of W_bef, BI of W_aft]
        fwd_imp_list (list): List of FI [FI of W_bef, FI of W_aft]
        n (int): Number of top n items to get (if None, get all items without sorting)
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
    
    # Determine indices
    if n is None:
        top_n_indices = np.arange(len(weighted_scores))  # All items without sorting
    else:
        top_n_indices = np.argsort(weighted_scores)[-n:][::-1]  # n items in descending order

    # Classify into bef and aft, restore to original shape
    top_n_bef = np.array([
        np.unravel_index(idx, shape_bef) for idx in top_n_indices if idx < split_idx
    ])
    top_n_aft = np.array([
        np.unravel_index(idx - split_idx, shape_aft) for idx in top_n_indices if idx >= split_idx
    ])

    return {"bef": top_n_bef, "aft": top_n_aft, "scores": weighted_scores[top_n_indices]}


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

    # Classify into bef and aft, restore to original shape
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
        batched_labels (list): Correct labels for each batch
        vit_from_last_layer (ViTFromLastLayer): Final layer wrapper of ViT model
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
            # print(f"cs: {cs.shape}") # shape: (32, 768)
            oi_expanded = cs.unsqueeze(1)
            # print(f"oi_expanded: {oi_expanded.shape}")  # shape: (32, 1, 768)
            # print(f"tgt_weight_expanded: {tgt_weight_expanded.shape}")  # shape: (1, 3072, 768)
            # Calculate on GPU until **
            impact_out_weight = tgt_weight_expanded * oi_expanded # shape: (32, 3072, 768)
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
