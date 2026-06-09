"""
exp-ig-1-1.py  —  IG localization time measurement

Measures wall-clock time for IG-based localization:
  full-model forward (all 12 layers), batch_size=1,
  num_points=20 scaled steps per sample

Operates on the repair-split target misclassification samples.

Usage:
    python exp-ig-1-1.py c100 0 1 src_tgt
    python exp-ig-1-1.py c100 0 1 tgt --fpfn fp
    python exp-ig-1-1.py tiny-imagenet 0 1 src_tgt
"""
import os, time, json
from tqdm import tqdm
import argparse
import numpy as np
import torch
from datasets import load_from_disk
from transformers import ViTForImageClassification

from utils.helper import get_device
from utils.vit_util import (
    transforms, transforms_c100,
    identfy_tgt_misclf,
    get_ori_model_predictions,
    maybe_initialize_repair_weights_,
)
from utils.constant import ViTExperiment

NUM_POINTS     = ViTExperiment.NUM_POINTS  # 20
TGT_LAYER      = 11   # last transformer layer (0-indexed)
IG_START_LAYER = 11   # last layer only
TGT_SPLIT      = "repair"


def scaled_input(emb: torch.Tensor, num_points: int):
    """Linearly interpolate from zeros to emb in num_points steps."""
    baseline = torch.zeros_like(emb)
    step = (emb - baseline) / num_points
    res = torch.cat([torch.add(baseline, step * i) for i in range(num_points)], dim=0)
    return res, step[0]


def time_ig(model, cached_mid_states_per_layer, pixel_values_list,
            tgt_label, device, num_points, ig_layers):
    """
    Time IG computation, faithfully following calc_integrated_gradient.py:
      - outer loop: ig_layers
      - inner loop: samples one by one (batch_size=1 in sample space)
      - per sample: scaled_input (num_points steps) → full model forward → grad.sum

    Returns:
        elapsed seconds (float)
    """
    tgt_pos = ViTExperiment.CLS_IDX
    model.eval()

    start = time.perf_counter()
    for tgt_layer in ig_layers:
        grad_list = []
        cached_mid_states = cached_mid_states_per_layer[tgt_layer]

        for data_idx, pixel_values in tqdm(enumerate(pixel_values_list), total=len(pixel_values_list), desc=f"IG layer={tgt_layer}"):
            x = pixel_values.to(device)
            tgt_mid = cached_mid_states[data_idx].unsqueeze(0).to(device)
            scaled_weights, _ = scaled_input(tgt_mid, num_points)
            scaled_weights.requires_grad_(True)

            output = model(
                x,
                tgt_pos=tgt_pos,
                tgt_layer=tgt_layer,
                tmp_score=scaled_weights,
                tgt_label=tgt_label,
            )
            grad = output.gradients
            grad = grad.sum(dim=0)
            grad_list.append(grad.detach().cpu())
    end = time.perf_counter()
    return end - start


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ds",          type=str)
    parser.add_argument("k",           type=int)
    parser.add_argument("tgt_rank",    type=int)
    parser.add_argument("misclf_type", type=str, choices=["src_tgt", "tgt"])
    parser.add_argument("--fpfn",           type=str, default=None, choices=["fp", "fn"])
    parser.add_argument("--n_reps",         type=int, default=1)
    parser.add_argument("--ig_start_layer", type=int, default=IG_START_LAYER)
    args = parser.parse_args()

    ds_name     = args.ds
    k           = args.k
    tgt_rank    = args.tgt_rank
    misclf_type = args.misclf_type
    fpfn        = args.fpfn
    n_reps      = args.n_reps
    ig_start    = args.ig_start_layer
    ig_layers   = list(range(ig_start, TGT_LAYER + 1))

    device = get_device()
    pretrained_dir = getattr(ViTExperiment, ds_name.replace("-", "_")).OUTPUT_DIR.format(k=k)

    # ── Dataset ──────────────────────────────────────────────────────────────
    if ds_name == "tiny-imagenet":
        tf_func   = transforms
        label_col = "label"
    elif ds_name == "c100":
        tf_func   = transforms_c100
        label_col = "fine_label"
    else:
        raise NotImplementedError(ds_name)

    ds = load_from_disk(os.path.join(ViTExperiment.DATASET_DIR, f"{ds_name}_fold{k}"))
    labels = {split: np.array(ds[split][label_col]) for split in ("train", "repair", "test")}

    # ── Misclassification info ────────────────────────────────────────────────
    misclf_info_dir = os.path.join(pretrained_dir, "misclf_info")
    misclf_pair, tgt_label, tgt_mis_indices = identfy_tgt_misclf(
        misclf_info_dir, tgt_split=TGT_SPLIT, tgt_rank=tgt_rank,
        misclf_type=misclf_type, fpfn=fpfn,
    )
    ig_tgt_label = tgt_label if tgt_label is not None else misclf_pair[1]
    print(f"[INFO] tgt_label: {tgt_label}, ig_tgt_label: {ig_tgt_label}")

    pred_res_dir = os.path.join(pretrained_dir, "pred_results", "PredictionOutput")
    if misclf_type == "tgt":
        _, _, indices_cor_tgt, _, indices_cor_others = get_ori_model_predictions(
            pred_res_dir, labels, tgt_split=TGT_SPLIT, misclf_type=misclf_type, tgt_label=tgt_label
        )
        indices_to_correct = np.sort(np.concatenate([indices_cor_tgt, indices_cor_others]))
    else:
        _, _, indices_to_correct = get_ori_model_predictions(
            pred_res_dir, labels, tgt_split=TGT_SPLIT, misclf_type=misclf_type, tgt_label=tgt_label
        )

    print(f"[INFO] tgt_mis_indices:    {len(tgt_mis_indices)}")
    print(f"[INFO] indices_to_correct: {len(indices_to_correct)}")
    ig_sample_indices = np.concatenate([indices_to_correct, tgt_mis_indices])
    n_ig_samples = len(ig_sample_indices)

    print(f"[INFO] IG layers: {ig_layers}")
    print(f"[INFO] IG: {len(ig_layers)} layer(s) × {n_ig_samples} samples × {NUM_POINTS} steps "
          f"= {len(ig_layers)*n_ig_samples*NUM_POINTS} full-model forward passes")

    # ── Pixel values for IG ──────────────────────────────────────────────────
    repair_ds_preprocessed = ds[TGT_SPLIT].with_transform(tf_func)
    pixel_values_list = []
    for idx in ig_sample_indices:
        pv = repair_ds_preprocessed[int(idx)]["pixel_values"]
        if not isinstance(pv, torch.Tensor):
            pv = torch.tensor(pv)
        pixel_values_list.append(pv.unsqueeze(0))

    # ── Cached intermediate states for IG ───────────────────────────────────
    cached_mid_states_per_layer = {}
    for layer_idx in ig_layers:
        mid_cache_path = os.path.join(
            pretrained_dir, f"cache_states_{TGT_SPLIT}",
            f"intermediate_states_l{layer_idx}.pt"
        )
        assert os.path.exists(mid_cache_path), f"Not found: {mid_cache_path}"
        cached_all = torch.load(mid_cache_path, map_location="cpu")
        cached_mid_states_per_layer[layer_idx] = cached_all[ig_sample_indices]

    # ── Load model ───────────────────────────────────────────────────────────
    model, loading_info = ViTForImageClassification.from_pretrained(
        pretrained_dir, output_loading_info=True
    )
    model.to(device).eval()
    model = maybe_initialize_repair_weights_(model, loading_info["missing_keys"])

    # ── Timing ───────────────────────────────────────────────────────────────
    ig_times = []
    for rep in range(n_reps):
        print(f"\n--- Rep {rep+1}/{n_reps} ---")
        t_ig = time_ig(
            model, cached_mid_states_per_layer, pixel_values_list,
            ig_tgt_label, device, NUM_POINTS, ig_layers
        )
        print(f"  IG: {t_ig:.2f} sec")
        ig_times.append(t_ig)

    # ── Results ──────────────────────────────────────────────────────────────
    result = {
        "ds": ds_name, "k": k, "tgt_rank": tgt_rank,
        "misclf_type": misclf_type, "fpfn": fpfn,
        "n_tgt_mis":   int(len(tgt_mis_indices)),
        "n_correct":   int(len(indices_to_correct)),
        "n_ig_samples": n_ig_samples,
        "num_points":  NUM_POINTS,
        "ig_layers":   ig_layers,
        "ig_total_forward_passes": int(len(ig_layers) * n_ig_samples * NUM_POINTS),
        "ig_time_sec": float(ig_times[0]),
    }

    print(f"\n{'='*60}")
    print(f"  IG: {result['ig_time_sec']:.2f} sec")
    print(f"{'='*60}")

    misclf_ptn   = misclf_type if fpfn is None else f"{misclf_type}_{fpfn}"
    ig_layer_str = f"l{ig_layers[0]}tol{ig_layers[-1]}"
    out_path = os.path.join(
        pretrained_dir,
        f"misclf_top{tgt_rank}",
        f"exp-ig-1-1_{ds_name}_k{k}_top{tgt_rank}_{misclf_ptn}_{ig_layer_str}.json"
    )
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"[INFO] Saved to {out_path}")
