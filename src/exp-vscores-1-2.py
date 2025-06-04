import os, sys, time
import argparse
from tqdm import tqdm
import numpy as np
import torch
from datasets import load_from_disk
from transformers import ViTForImageClassification
from utils.helper import get_device
from utils.vit_util import transforms, transforms_c100, get_vscore, maybe_initialize_repair_weights_
from utils.constant import ViTExperiment

if __name__ == "__main__":
    # データセットをargparseで受け取る
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str)
    parser.add_argument('k', type=int, help="the fold id (0 to K-1)")
    parser.add_argument('--take_abs', action="store_true", default=False, help="take absolute value of vscore")
    parser.add_argument('--take_covavg', action="store_true", default=False, help="take average value of Cov for vscore")
    args = parser.parse_args()
    ds_name = args.ds
    k = args.k
    abs = args.take_abs
    covavg = args.take_covavg
    print(f"ds_name: {ds_name}, fold_id: {k}, abs: {abs}, covavg: {covavg}")

    # datasetごとに違う変数のセット
    if ds_name == "c10" or ds_name == "tiny-imagenet":
        tf_func = transforms
        label_col = "label"
    elif ds_name == "c100":
        tf_func = transforms_c100
        label_col = "fine_label"
    else:
        NotImplementedError
        
    dataset_dir = ViTExperiment.DATASET_DIR
    exp_obj = getattr(ViTExperiment, ds_name.replace("-", "_"))
    ds = load_from_disk(os.path.join(dataset_dir, f"{ds_name}_fold{k}"))
    pretrained_dir = exp_obj.OUTPUT_DIR.format(k=k)

    tgt_pos = ViTExperiment.CLS_IDX
    # デバイス (cuda, or cpu) の取得
    device = get_device()
    split_names = list(ds.keys()) # [train, repair, test]
    target_split_names = ["repair"]
    # target_split_namesは全てsplit_namesに含まれていることを前提とする
    assert all([split_name in split_names for split_name in target_split_names]), f"target_split_names should be subset of split_names"
    # ラベルの取得
    labels = {
        "train": np.array(ds["train"][label_col]),
        "repair": np.array(ds["repair"][label_col]),
        "test": np.array(ds["test"][label_col])
    }
    # 読み込まれた時にリアルタイムで前処理を適用するようにする
    ds_preprocessed = ds.with_transform(tf_func)
    # pretrained modelのロード
    model, loading_info = ViTForImageClassification.from_pretrained(pretrained_dir, output_loading_info=True)
    model.to(device).eval()
    model = maybe_initialize_repair_weights_(model, loading_info["missing_keys"])
    model.eval()
    end_li = model.vit.config.num_hidden_layers
    batch_size = ViTExperiment.BATCH_SIZE

    for split in target_split_names:
        print(f"Process {split}...")
        # 必要なディレクトリがない場合は先に作っておく
        vscore_dir = os.path.join(pretrained_dir, "vscores")
        os.makedirs(vscore_dir, exist_ok=True)
        cache_dir = os.path.join(pretrained_dir, f"cache_states_{split}")
        os.makedirs(cache_dir, exist_ok=True)
        all_mid_states = []
        all_logits = []
        # loop for dataset batch
        for entry_dic in tqdm(ds_preprocessed[split].iter(batch_size=batch_size), total=len(ds_preprocessed[split])//batch_size+1):
            x, y = entry_dic["pixel_values"].to(device), entry_dic["labels"]
            output = model.forward(x, output_intermediate_states=True, tgt_pos=tgt_pos)
            # CLSトークンに対応するintermediate statesを取得
            output_mid_states = np.array([mid_states_each_layer[1] for mid_states_each_layer in output.intermediate_states])
            output_mid_states = output_mid_states.transpose(1, 0, 2) # (batch_size, num_layers, num_neurons)
            logits = output.logits.cpu().detach().numpy()
            all_logits.append(logits)
            all_mid_states.append(output_mid_states)
        all_logits = np.concatenate(all_logits) # (num_samples, num_labels)
        all_pred_labels = np.argmax(all_logits, axis=-1) # (num_samples, )
        all_mid_states = np.concatenate(all_mid_states) # (num_samples, num_layers, num_neurons)
        true_labels = labels[split] # (num_samples, )
        # サンプルごとの予測の正解不正解の配列を作る
        is_correct = all_pred_labels == true_labels
        
        unique_labels = np.unique(true_labels)
        for label_id in unique_labels:
            print(f"Processing label: {label_id}")
            label_mask = true_labels == label_id
            correct_mask = is_correct & label_mask
            incorrect_mask = ~is_correct & label_mask

            label_correct_states = all_mid_states[correct_mask]
            label_incorrect_states = all_mid_states[incorrect_mask]

            for cor_mis, mid_states in zip(["cor", "mis"], [label_correct_states, label_incorrect_states]):
                print(f"  [{cor_mis}] samples: {len(mid_states)}")
                vscore_per_layer = []
                for tgt_layer in range(end_li):
                    tgt_mid_state = mid_states[:, tgt_layer, :]
                    vscore = get_vscore(tgt_mid_state, abs=abs, covavg=covavg)
                    vscore_per_layer.append(vscore)
                vscores = np.array(vscore_per_layer)

                prefix = ("vscore_abs" if abs else "vscore") + ("_covavg" if covavg else "")
                save_name = f"{prefix}_l1tol{end_li}_label_{label_id}_ori_{split}_{cor_mis}.npy"
                vscore_save_path = os.path.join(vscore_dir, save_name)
                np.save(vscore_save_path, vscores)
                print(f"  → vscore saved at {vscore_save_path}")