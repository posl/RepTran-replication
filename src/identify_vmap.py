import os
from collections import defaultdict
import numpy as np
import argparse
from transformers import ViTForImageClassification
from utils.constant import ViTExperiment
from utils.helper import get_device, get_corruption_types


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str)
    parser.add_argument('--used_column', type=str, default="train")
    parser.add_argument("--start_layer_idx", type=int, default=0)
    args = parser.parse_args()
    ds_name = args.ds
    start_li = args.start_layer_idx
    used_column = args.used_column
    # argparseで受け取った引数のサマリーを表示
    print(f"ds_name: {ds_name}, start_layer_idx: {start_li}, used_column: {used_column}")
    # corruption typeのリストとデバイス設定
    ct_list = get_corruption_types()
    device = get_device()
    # Set different variables for each dataset
    if ds_name == "c10":
        label_col = "label"
        num_labels = 10
    elif ds_name == "c100":
        label_col = "fine_label"
        num_labels = 100
    else:
        NotImplementedError
    # Load pretrained model
    pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR
    model = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
    model.eval()
    # 対象の設定
    end_li = model.vit.config.num_hidden_layers
    ct_list = get_corruption_types()
    target_layer = 11 # vmap.ipynbの調査結果から，最終層だけ対象にする

    vscore_dir = os.path.join(getattr(ViTExperiment, ds_name).OUTPUT_DIR, "neuron_scores")
    vmap_dic = defaultdict(defaultdict)
    for tgt_ct in ct_list:
        print(f"\ntarget corruption type: {tgt_ct}\n")
        vmap_dic[tgt_ct] = defaultdict(defaultdict)
        vmap_dir = os.path.join(getattr(ViTExperiment, ds_name).OUTPUT_DIR, f"{tgt_ct}_severity4", "vmap") # 結果Save先のdir
        os.makedirs(vmap_dir, exist_ok=True)
        for cor_mis in ["cor", "mis"]:
            vmap_dic[tgt_ct][cor_mis] = defaultdict(np.array)
            print(f"Cor or Mis: {cor_mis}")
            ds_type = f"{tgt_ct}_{used_column}"
            vscore_save_path = os.path.join(vscore_dir, f"vscore_l{start_li}tol{end_li}_all_label_{ds_type}_{cor_mis}.npy")
            # vscoreをロード
            vscores = np.load(vscore_save_path)
            vmap_dic[tgt_ct][cor_mis] = vscores.T
            print(f"vscores shape: {vmap_dic[tgt_ct][cor_mis].shape}")
        # 正解時と不正解時のVmapのDiffを計算
        vmap_cor = vmap_dic[tgt_ct]["cor"]
        vmap_mis = vmap_dic[tgt_ct]["mis"]
        vmap_diff = vmap_cor - vmap_mis
        vmap_dic[tgt_ct]["diff"] = vmap_diff
        # top10%のニューロンのindexを取得
        top10 = np.percentile(np.abs(vmap_diff[:, target_layer]), 90)
        condition = np.abs(vmap_diff[:, target_layer]).reshape(-1) > top10
        top10_idx = np.where(condition)[0]
        print(f"Top 10% Neurons (Layer {target_layer}, {len(top10_idx)} neurons): {top10_idx}")
        # vmap_diffとtop10%のニューロンの位置をSave
        vmap_diff_path = os.path.join(vmap_dir, f"vmap_diff_{used_column}_l{start_li}tol{end_li}.npy")
        np.save(vmap_diff_path, vmap_diff)
        top10_idx_path = os.path.join(vmap_dir, f"top10_idx_{used_column}_l{target_layer}.npy")
        np.save(top10_idx_path, top10_idx)