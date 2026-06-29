"""
exp-repair-7-1-1.py  —  Meta3 / RQ6: neuron-score component ablation (runner)

Equal-budget (N_w=472) ablation of the neuron score used in our FL.
The neuron score is composed of two min-max normalized components:
    neuron_score = VDiff x MisAct
This runner recomputes localization with one component ablated and then runs DE repair:
    --score_mode vdiff   : VDiff only  (MisAct treated as a uniform 1)
    --score_mode misact  : MisAct only (VDiff  treated as a uniform 1)
The two existing conditions (Full=ours, No-neuron-score=bl) are reused from exp-repair-4-1
and are NOT re-run here.

Fixed: N_w=472, alpha=10/11 (Arachne default), bounds=Arachne, p=default(0.5/0.5)
       (p is a weight-level knob orthogonal to this RQ, kept at the exp-repair-4 default).

Usage:
    python exp-repair-7-1-1.py c100 0 1 0 --score_mode vdiff  --misclf_type src_tgt
    python exp-repair-7-1-1.py c100 0 1 0 --score_mode misact --misclf_type tgt --fpfn fp
"""
import os, time, json
import argparse
import numpy as np
import torch
import torch.optim as optim
from datasets import load_from_disk
from transformers import ViTForImageClassification

from utils.helper import get_device
from utils.vit_util import (
    transforms, transforms_c100,
    ViTFromLastLayer,
    identfy_tgt_misclf,
    get_ori_model_predictions,
    get_new_model_predictions,
    get_batched_hs,
    get_batched_labels,
    localize_neurons_with_mean_activation,
    sample_true_positive_indices_per_class,
    maybe_initialize_repair_weights_,
)
from utils.arachne import calculate_bi_fi, calculate_top_n_flattened
from utils.constant import ViTExperiment
from utils.log import set_exp_logging
from utils.de import DE_searcher
from logging import getLogger

logger = getLogger("base_logger")

# ── Fixed hyperparameters ────────────────────────────────────────────────────
FIXED_WNUM       = 472
ALPHA_IN_ARACHNE = 10
FIXED_ALPHA      = ALPHA_IN_ARACHNE / (1 + ALPHA_IN_ARACHNE)  # ≈ 0.9091
FIXED_BOUNDS     = "Arachne"
RATIO_SAMPLED_FROM_CORRECT = 0.05
MAX_SEARCH_NUM   = 50
POP_SIZE         = 100
TGT_SPLIT        = "repair"
TGT_LAYER        = 11


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ds",       type=str)
    parser.add_argument("k",        type=int,   help="fold id")
    parser.add_argument("tgt_rank", type=int,   help="rank of target misclassification")
    parser.add_argument("reps_id",  type=int,   help="repetition id")
    parser.add_argument("--score_mode", type=str, required=True, choices=["vdiff", "misact"],
                        help="neuron-score component to keep (vdiff=VDiff only, misact=MisAct only)")
    parser.add_argument("--misclf_type", type=str, default="tgt", choices=["src_tgt", "tgt"])
    parser.add_argument("--fpfn",      type=str,   default=None, choices=["fp", "fn"])
    parser.add_argument("--wnum",      type=int,   default=FIXED_WNUM,
                        help="equal weight budget N_w (default 472; use 236 for the tight-budget ablation)")
    args = parser.parse_args()

    ds_name     = args.ds
    k           = args.k
    tgt_rank    = args.tgt_rank
    reps_id     = args.reps_id
    score_mode  = args.score_mode
    misclf_type = args.misclf_type
    fpfn        = args.fpfn
    wnum        = args.wnum

    setting_id = f"n{wnum}_alpha{FIXED_ALPHA}_bounds{FIXED_BOUNDS}_{score_mode}_ours"

    pretrained_dir = getattr(ViTExperiment, ds_name.replace("-", "_")).OUTPUT_DIR.format(k=k)

    # ── Save directories ─────────────────────────────────────────────────────
    if fpfn is not None and misclf_type == "tgt":
        location_save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_{fpfn}_weights_location")
        save_dir          = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_{fpfn}_repair_weight_by_de")
    elif misclf_type == "all":
        location_save_dir = os.path.join(pretrained_dir, "all_weights_location")
        save_dir          = os.path.join(pretrained_dir, "all_repair_weight_by_de")
    else:
        location_save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_weights_location")
        save_dir          = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_repair_weight_by_de")
    os.makedirs(save_dir, exist_ok=True)

    # ── File paths ───────────────────────────────────────────────────────────
    patch_save_path       = os.path.join(save_dir, f"exp-repair-7-1-best_patch_{setting_id}_reps{reps_id}.npy")
    tgt_indices_save_path = os.path.join(save_dir, f"exp-repair-7-1-tgt_indices_{setting_id}_reps{reps_id}.npy")
    metrics_json_path     = os.path.join(save_dir, f"exp-repair-7-1-metrics_for_repair_{setting_id}_reps{reps_id}.json")
    tracker_save_path     = os.path.join(save_dir, f"exp-repair-7-1-tracker_{setting_id}_reps{reps_id}.pkl")

    # ── Logger ───────────────────────────────────────────────────────────────
    logger = set_exp_logging(exp_dir=save_dir, exp_name=f"exp-repair-7-1-1_{setting_id}_reps{reps_id}")
    logger.info(f"ds={ds_name}, k={k}, tgt_rank={tgt_rank}, reps_id={reps_id}, "
                f"score_mode={score_mode}, misclf_type={misclf_type}, fpfn={fpfn}")
    logger.info(f"setting_id={setting_id}, FIXED_ALPHA={FIXED_ALPHA:.6f} "
                f"(alpha_in_arachne={ALPHA_IN_ARACHNE}), wnum={wnum}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    if ds_name in ("c10", "tiny-imagenet"):
        tf_func   = transforms
        label_col = "label"
    elif ds_name == "c100":
        tf_func   = transforms_c100
        label_col = "fine_label"
    else:
        raise NotImplementedError(ds_name)

    ds = load_from_disk(os.path.join(ViTExperiment.DATASET_DIR, f"{ds_name}_fold{k}"))
    labels = {split: np.array(ds[split][label_col]) for split in ("train", "repair", "test")}
    device     = get_device()
    batch_size = ViTExperiment.BATCH_SIZE
    tgt_pos    = ViTExperiment.CLS_IDX

    # ── Model ─────────────────────────────────────────────────────────────────
    model, loading_info = ViTForImageClassification.from_pretrained(
        pretrained_dir, output_loading_info=True
    )
    model.to(device).eval()
    model = maybe_initialize_repair_weights_(model, loading_info["missing_keys"])

    # ── Misclassification indices ──────────────────────────────────────────────
    misclf_info_dir = os.path.join(pretrained_dir, "misclf_info")
    misclf_pair, tgt_label, tgt_mis_indices = identfy_tgt_misclf(
        misclf_info_dir, tgt_split=TGT_SPLIT, tgt_rank=tgt_rank,
        misclf_type=misclf_type, fpfn=fpfn,
    )
    pred_res_dir = os.path.join(pretrained_dir, "pred_results", "PredictionOutput")
    if misclf_type == "tgt":
        ori_pred_labels, _, indices_cor_tgt, _, indices_cor_others = get_ori_model_predictions(
            pred_res_dir, labels, tgt_split=TGT_SPLIT, misclf_type=misclf_type, tgt_label=tgt_label
        )
        indices_to_correct = np.sort(np.concatenate([indices_cor_tgt, indices_cor_others]))
    else:
        ori_pred_labels, _, indices_to_correct = get_ori_model_predictions(
            pred_res_dir, labels, tgt_split=TGT_SPLIT, misclf_type=misclf_type, tgt_label=tgt_label
        )
    print(f"len(indices_to_correct): {len(indices_to_correct)}, len(tgt_mis_indices): {len(tgt_mis_indices)}")

    # ===============================================
    # Localization phase
    # ===============================================
    # score_mode controls the neuron-score composition → must recompute per score_mode
    logger.info(f"Computing localization inline with score_mode={score_mode}")

    # Load cached intermediate neuron values (for neuron_scores)
    mid_cache_dir  = os.path.join(pretrained_dir, f"cache_states_{TGT_SPLIT}")
    mid_save_path  = os.path.join(mid_cache_dir, f"intermediate_states_l{TGT_LAYER}.pt")
    cached_mid_states = torch.load(mid_save_path, map_location="cpu").detach().numpy().copy()
    print(f"cached_mid_states.shape: {cached_mid_states.shape}")

    # vscore dirs
    if misclf_type in ("src_tgt", "tgt"):
        vscore_before_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", "vscores_before")
        vscore_dir        = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", "vscores")
        vscore_after_dir  = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", "vscores_after")
    else:
        vscore_before_dir = os.path.join(pretrained_dir, "vscores_before")
        vscore_dir        = os.path.join(pretrained_dir, "vscores")
        vscore_after_dir  = os.path.join(pretrained_dir, "vscores_after")
    vscore_cor_dir = os.path.join(pretrained_dir, "vscores")

    # Neuron scores with the selected component (VDiff-only or MisAct-only)
    _, _, neuron_scores = localize_neurons_with_mean_activation(
        vscore_before_dir, vscore_dir, vscore_after_dir, TGT_LAYER, n=None,
        intermediate_states=cached_mid_states,
        tgt_mis_indices=tgt_mis_indices,
        misclf_pair=misclf_pair,
        tgt_label=tgt_label,
        fpfn=fpfn,
        return_all_neuron_score=True,
        vscore_abs=True,
        covavg=False,
        vscore_cor_dir=vscore_cor_dir,
        score_mode=score_mode,
    )

    # BI/FI computation (ModFI + ModGL gradients)
    cache_path = os.path.join(
        pretrained_dir, f"cache_hidden_states_before_layernorm_{TGT_SPLIT}",
        f"hidden_states_before_layernorm_{TGT_LAYER}.npy"
    )
    assert os.path.exists(cache_path), f"{cache_path} does not exist."

    correct_batched_hs     = get_batched_hs(cache_path, batch_size, indices_to_correct)
    correct_batched_labels = get_batched_labels(labels[TGT_SPLIT], batch_size, indices_to_correct)
    incorrect_batched_hs   = get_batched_hs(cache_path, batch_size, tgt_mis_indices)
    incorrect_batched_labels = get_batched_labels(labels[TGT_SPLIT], batch_size, tgt_mis_indices)

    vit_from_last_layer = ViTFromLastLayer(model)
    vit_from_last_layer.eval()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    print("Calculating BI and FI... (correct samples)")
    pos_results = calculate_bi_fi(
        indices_to_correct, correct_batched_hs, correct_batched_labels,
        vit_from_last_layer, optimizer, tgt_pos,
    )
    print("Calculating BI and FI... (incorrect samples)")
    neg_results = calculate_bi_fi(
        tgt_mis_indices, incorrect_batched_hs, incorrect_batched_labels,
        vit_from_last_layer, optimizer, tgt_pos,
    )

    grad_loss_list, fwd_imp_list = [], []
    for ba in ["before", "after"]:
        grad_loss = neg_results[ba]["bw"] / (1 + pos_results[ba]["bw"])
        fwd_imp   = neg_results[ba]["fw"] / (1 + pos_results[ba]["fw"])
        grad_loss_list.append(grad_loss)
        fwd_imp_list.append(fwd_imp)

    # Adjust with neuron_scores (ablated component)
    for i in range(2):
        broadcasted = neuron_scores[:, np.newaxis] if i == 0 else neuron_scores[np.newaxis, :]
        grad_loss_list[i] *= broadcasted
        fwd_imp_list[i]   *= broadcasted

    # Top-N selection with default p (weight_grad_loss=weight_fwd_imp=0.5), matching exp-repair-4
    identified = calculate_top_n_flattened(
        grad_loss_list, fwd_imp_list, n=wnum,
    )
    pos_before = identified["bef"]
    pos_after  = identified["aft"]

    # Save score_mode-specific location file for use by evaluation script (7-1-3.py)
    loc_save_path = os.path.join(
        location_save_dir,
        f"exp-repair-7-1-location_n{wnum}_{score_mode}_weight_ours.npy"
    )
    np.save(loc_save_path, (pos_before, pos_after))
    logger.info(f"Saved score_mode-specific location to {loc_save_path}")

    logger.info(f"pos_before={pos_before}")
    logger.info(f"pos_after={pos_after}")
    logger.info(f"num(pos_to_fix)={len(pos_before)}+{len(pos_after)}={len(pos_before)+len(pos_after)}")
    print(f"num(pos_to_fix)={len(pos_before)}+{len(pos_after)}={len(pos_before)+len(pos_after)}")

    # ── Sample correct indices for DE fitness ─────────────────────────────────
    ori_tgt_labels = labels[TGT_SPLIT]
    num_sampled    = int(RATIO_SAMPLED_FROM_CORRECT * len(ori_tgt_labels))
    sampled_correct = sample_true_positive_indices_per_class(
        num_sampled, indices_to_correct, ori_pred_labels,
    )
    tgt_indices = sampled_correct.tolist() + tgt_mis_indices.tolist()
    assert len(tgt_indices) == len(set(tgt_indices))
    np.save(tgt_indices_save_path, tgt_indices)
    logger.info(f"tgt_indices: len={len(tgt_indices)}")

    cache_path = os.path.join(
        pretrained_dir, f"cache_hidden_states_before_layernorm_{TGT_SPLIT}",
        f"hidden_states_before_layernorm_{TGT_LAYER}.npy"
    )
    batch_hs_tgt     = get_batched_hs(cache_path, batch_size, tgt_indices, device=device)
    batch_labels_tgt = get_batched_labels(ori_tgt_labels, batch_size, tgt_indices)

    ori_pred_tgt, ori_true_tgt = get_new_model_predictions(
        vit_from_last_layer, batch_hs_tgt, batch_labels_tgt, tgt_pos=tgt_pos,
    )
    ori_is_correct_tgt       = ori_pred_tgt == ori_true_tgt
    indices_correct_for_de   = np.arange(len(sampled_correct))
    indices_incorrect_for_de = np.arange(len(sampled_correct), len(tgt_indices))
    assert np.sum(ori_is_correct_tgt[indices_correct_for_de])   == len(indices_correct_for_de)
    assert np.sum(ori_is_correct_tgt[indices_incorrect_for_de]) == 0

    weight_before2med = vit_from_last_layer.base_model_last_layer.intermediate.dense.weight.cpu().detach().numpy()
    weight_med2after  = vit_from_last_layer.base_model_last_layer.output.dense.weight.cpu().detach().numpy()

    # ── DE search ─────────────────────────────────────────────────────────────
    searcher = DE_searcher(
        batch_hs_before_layernorm=batch_hs_tgt,
        batch_labels=batch_labels_tgt,
        indices_to_correct=indices_correct_for_de,
        indices_to_wrong=indices_incorrect_for_de,
        num_label=len(set(labels["train"])),
        indices_to_target_layers=[TGT_LAYER],
        device=device,
        mutation=(0.5, 1),
        recombination=0.7,
        max_search_num=MAX_SEARCH_NUM,
        partial_model=vit_from_last_layer,
        alpha=FIXED_ALPHA,
        pop_size=POP_SIZE,
        mode="weight",
        pos_before=pos_before,
        pos_after=pos_after,
        weight_before2med=weight_before2med,
        weight_med2after=weight_med2after,
        custom_bounds=FIXED_BOUNDS,
    )

    logger.info("Start DE search...")
    s = time.perf_counter()
    searcher.search(
        patch_save_path=patch_save_path,
        pos_before=pos_before,
        pos_after=pos_after,
        tracker_save_path=tracker_save_path,
    )
    e = time.perf_counter()
    tot_time = e - s
    logger.info(f"Total execution time: {tot_time:.1f} sec.")
    print(f"Total execution time: {tot_time:.1f} sec.")

    with open(metrics_json_path, "w") as f:
        json.dump({"tot_time": tot_time}, f)
    logger.info(f"Metrics saved to {metrics_json_path}")
    print(f"===== Completed exp-repair-7-1-1.py (score_mode={score_mode}, reps={reps_id}) =====")
