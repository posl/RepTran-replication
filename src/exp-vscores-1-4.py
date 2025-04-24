import os, sys, time
import argparse
from tqdm import tqdm
import numpy as np
import torch
from datasets import load_from_disk
from transformers import ViTForImageClassification
from utils.helper import get_device
from utils.vit_util import transforms, transforms_c100, get_vscore
from utils.constant import ViTExperiment

if __name__ == "__main__":
    # --- argparse ---
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str)
    parser.add_argument('k', type=int, help="the fold id (0 to K-1)")
    parser.add_argument('--take_abs', action="store_true", default=False, help="take absolute value of vscore")
    parser.add_argument('--take_covavg', action="store_true", default=False, help="take average value of Cov for vscore")
    parser.add_argument('--min_samples', type=int, default=5, help="minimum #samples per class for subsampling")
    args = parser.parse_args()

    ds_name = args.ds
    k = args.k
    abs = args.take_abs
    covavg = args.take_covavg
    min_samples = args.min_samples
    print(f"ds_name: {ds_name}, fold_id: {k}, abs: {abs}, covavg: {covavg}, min_samples: {min_samples}")

    # --- dataset setup ---
    if ds_name == "c10":
        tf_func = transforms
        label_col = "label"
    elif ds_name == "c100":
        tf_func = transforms_c100
        label_col = "fine_label"
    else:
        raise NotImplementedError

    tgt_pos = ViTExperiment.CLS_IDX
    ds_dirname = f"{ds_name}_fold{k}"
    device = get_device()
    ds = load_from_disk(os.path.join(ViTExperiment.DATASET_DIR, ds_dirname))
    split_names = list(ds.keys())
    target_split_names = ["repair"]
    assert all([split in split_names for split in target_split_names])

    labels = {split: np.array(ds[split][label_col]) for split in split_names}
    ds_preprocessed = ds.with_transform(tf_func)

    pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR.format(k=k)
    model = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
    model.eval()
    end_li = model.vit.config.num_hidden_layers
    batch_size = ViTExperiment.BATCH_SIZE

    for split in target_split_names:
        print(f"Process {split}...")
        vscore_dir = os.path.join(pretrained_dir, "vscores_subsampled")
        os.makedirs(vscore_dir, exist_ok=True)

        # --- forward pass to collect mid-states and predictions ---
        all_mid_states = []
        all_logits = []
        for entry in tqdm(ds_preprocessed[split].iter(batch_size=batch_size), total=len(ds_preprocessed[split])//batch_size+1):
            x = entry["pixel_values"].to(device)
            output = model(x, output_intermediate_states=True, tgt_pos=tgt_pos)
            states = np.array([s[1] for s in output.intermediate_states])
            states = states.transpose(1, 0, 2)  # (B, L, H)
            logits = output.logits.cpu().detach().numpy()
            all_mid_states.append(states)
            all_logits.append(logits)

        all_mid_states = np.concatenate(all_mid_states)
        all_logits = np.concatenate(all_logits)
        true_labels = labels[split]
        pred_labels = np.argmax(all_logits, axis=1)
        is_correct = true_labels == pred_labels

        # --- subsample per label ---
        unique_labels = np.unique(true_labels)
        for label_id in unique_labels:
            label_mask = true_labels == label_id
            cor_idx = np.where(is_correct & label_mask)[0]
            mis_idx = np.where(~is_correct & label_mask)[0]
            n = min(len(cor_idx), len(mis_idx))
            if n < min_samples:
                print(f"[Skip] label {label_id} (not enough samples: cor={len(cor_idx)}, mis={len(mis_idx)})")
                continue

            sub_cor = np.random.choice(cor_idx, n, replace=False)
            sub_mis = np.random.choice(mis_idx, n, replace=False)

            for cor_mis, idx in zip(["cor", "mis"], [sub_cor, sub_mis]):
                print(f"  [label {label_id}][{cor_mis}] using {len(idx)} samples")
                vscore_per_layer = []
                for tgt_layer in range(end_li):
                    tgt_mid = all_mid_states[idx, tgt_layer, :]
                    vscore = get_vscore(tgt_mid, abs=abs, covavg=covavg)
                    vscore_per_layer.append(vscore)
                vscores = np.stack(vscore_per_layer)

                prefix = ("vscore_abs" if abs else "vscore") + ("_covavg" if covavg else "")
                save_name = f"{prefix}_l1tol{end_li}_label_{label_id}_ori_{split}_{cor_mis}.npy"
                save_path = os.path.join(vscore_dir, save_name)
                np.save(save_path, vscores)
                print(f"    -> saved at {save_path}")
