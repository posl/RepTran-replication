import os, sys, time
import numpy as np
import torch
import argparse
from datasets import load_from_disk
from transformers import ViTForImageClassification
from utils.helper import get_device
from utils.vit_util import transforms, transforms_c100, get_vscore
from utils.constant import ViTExperiment

if __name__ == "__main__":
    # プログラム引数の受け取り
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str)
    parser.add_argument('--used_column', type=str, default="train")
    parser.add_argument('--start_layer_idx', type=int, default=9)
    args = parser.parse_args()
    ds_name = args.ds
    start_layer_idx = args.start_layer_idx
    used_column = args.used_column
    # argparseで受け取った引数のサマリーを表示
    print(f"ds_name: {ds_name}, start_layer_idx: {start_layer_idx}, used_column: {used_column}")
    # Set different variables for each dataset
    if ds_name == "c10":
        tf_func = transforms
        label_col = "label"
        num_labels = 10
    elif ds_name == "c100":
        tf_func = transforms_c100
        label_col = "fine_label"
        num_labels = 100
    else:
        NotImplementedError

    # Get device (cuda or cpu)
    device = get_device()
    # Load dataset (takes some time only on first load)
    ds = load_from_disk(os.path.join(ViTExperiment.DATASET_DIR, ds_name))[used_column]
    # ラベルのリスト取得
    labels = np.array(ds[label_col])
    ds = ds.with_transform(tf_func)
    # Load pretrained model
    pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR
    model = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
    model.eval()
    # 対象レイヤの設定
    start_li = start_layer_idx
    end_li = model.vit.config.num_hidden_layers

    tic = time.perf_counter()
    
    # cached_intermediateのディレクトリ
    im_dir = os.path.join(getattr(ViTExperiment, ds_name).OUTPUT_DIR, f"cache_states_{used_column}")
    for tgt_label in range(num_labels):
        # labelsがtgt_labelのindexを取得
        tgt_idx = np.where(labels == tgt_label)[0]
        vscore_per_layer = []
        for tgt_layer in range(start_li, end_li):
            print(f"tgt_label={tgt_label}, tgt_layer={tgt_layer}")
            # 対象レイヤのcached_intermediateをロード
            im_states = torch.load(os.path.join(im_dir, f"intermediate_states_l{tgt_layer}.pt"))
            # numpyに変換しておく
            im_states = im_states.cpu().numpy() # (num_samples, num_neurons_of_tgt_layer)

            # tgt_labelに対するim_statesだけを取得
            tgt_im = im_states[tgt_idx] # (num_samples_for_tgt_label, num_neurons_of_tgt_layer)
            # vscoreを計算
            vscore = get_vscore(tgt_im)
            vscore_per_layer.append(vscore)
        # vscoreをSave
        vscores = np.array(vscore_per_layer)
        # Save場所
        result_dir = os.path.join(getattr(ViTExperiment, ds_name).OUTPUT_DIR, "neuron_scores")
        vscore_save_path = os.path.join(result_dir, f"vscore_l{start_li}tol{end_li}_{tgt_label}.npy")
        np.save(vscore_save_path, vscores)
        print(f"vscore ({vscores.shape}) saved at {vscore_save_path}")
        
    toc = time.perf_counter()
    print(f"***** Costing time: {toc - tic:0.4f} seconds *****")