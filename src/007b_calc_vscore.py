"""
特定の誤分類タイプに対するVdiffの大きいニューロンを取得
"""

import os, sys, time
import argparse
import pickle
from tqdm import tqdm
import numpy as np
import torch
from datasets import load_from_disk
from transformers import ViTForImageClassification
from utils.helper import get_device
from utils.vit_util import transforms, transforms_c100, get_vscore, identfy_tgt_misclf
from utils.constant import ViTExperiment

if __name__ == "__main__":
    # データセットをargparseで受け取る
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str)
    parser.add_argument('k', type=int, help="the fold id (0 to K-1)")
    parser.add_argument('tgt_rank', type=int, help="the rank of the target misclassification type")
    parser.add_argument('--misclf_type', type=str, help="the type of misclassification (src_tgt or tgt)", default="tgt")
    parser.add_argument("--fpfn", type=str, help="the type of misclassification (fp or fn)", default=None, choices=["fp", "fn"])
    
    args = parser.parse_args()
    ds_name = args.ds
    k = args.k
    tgt_rank = args.tgt_rank
    misclf_type = args.misclf_type
    fpfn = args.fpfn
    print(f"ds_name: {ds_name}, fold_id: {k}, tgt_rank: {tgt_rank}, misclf_type: {misclf_type}, fpfn: {fpfn}")

    # datasetごとに違う変数のセット
    if ds_name == "c10":
        tf_func = transforms
        label_col = "label"
    elif ds_name == "c100":
        tf_func = transforms_c100
        label_col = "fine_label"
    else:
        NotImplementedError

    tgt_pos = ViTExperiment.CLS_IDX
    ds_dirname = f"{ds_name}_fold{k}"
    # デバイス (cuda, or cpu) の取得
    device = get_device()
    # datasetをロード (初回の読み込みだけやや時間かかる)
    ds = load_from_disk(os.path.join(ViTExperiment.DATASET_DIR, ds_dirname))
    tgt_split_names = list(ds.keys()) # [train, repair, test]
    # target_tgt_split_names = ["train", "repair"]
    target_tgt_split_names = ["repair"]
    # target_tgt_split_namesは全てtgt_split_namesに含まれていることを前提とする
    assert all([tgt_split_name in tgt_split_names for tgt_split_name in target_tgt_split_names]), f"target_tgt_split_names should be subset of tgt_split_names"
    # ラベルの取得
    labels = {
        "train": np.array(ds["train"][label_col]),
        "repair": np.array(ds["repair"][label_col]),
        "test": np.array(ds["test"][label_col])
    }
    # 読み込まれた時にリアルタイムで前処理を適用するようにする
    ds_preprocessed = ds.with_transform(tf_func)
    # pretrained modelのロード
    pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR.format(k=k)
    model = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
    model.eval()
    end_li = model.vit.config.num_hidden_layers
    batch_size = ViTExperiment.BATCH_SIZE
    
    # tgt_rankの誤分類情報を取り出す
    tgt_split = "repair"
    misclf_info_dir = os.path.join(pretrained_dir, "misclf_info")
    misclf_pair, tgt_label, tgt_mis_indices = identfy_tgt_misclf(misclf_info_dir, tgt_split=tgt_split, tgt_rank=tgt_rank, misclf_type=misclf_type, fpfn=fpfn)
    print(f"misclf_pair: {misclf_pair}, tgt_label: {tgt_label}, len(tgt_mis_indices): {len(tgt_mis_indices)}")
    if misclf_type == "src_tgt":
        slabel, tlabel = misclf_pair
    elif misclf_type == "tgt":
        tlabel = tgt_label

    print(f"Process {tgt_split}...")
    # 必要なディレクトリがない場合は先に作っておく
    vscore_before_dir = os.path.join(getattr(ViTExperiment, ds_name).OUTPUT_DIR.format(k=k), f"misclf_top{tgt_rank}", "vscores_before")
    vscore_after_dir = os.path.join(getattr(ViTExperiment, ds_name).OUTPUT_DIR.format(k=k), f"misclf_top{tgt_rank}", "vscores_after")
    vscore_med_dir = os.path.join(getattr(ViTExperiment, ds_name).OUTPUT_DIR.format(k=k), f"misclf_top{tgt_rank}", "vscores")
    os.makedirs(vscore_before_dir, exist_ok=True)
    os.makedirs(vscore_after_dir, exist_ok=True)
    os.makedirs(vscore_med_dir, exist_ok=True)
    # 前提: すでに全データに対するv-scoreが計算済みであること
    # なので，正しい分類ができたデータのv-scoreは計算せずコピーしてくる
    for vname in ["vscores_before", "vscores_after", "vscores"]:
        cor_vscore_path = os.path.join(getattr(ViTExperiment, ds_name).OUTPUT_DIR.format(k=k), vname, f"vscore_l1tol{end_li}_all_label_ori_{tgt_split}_cor.npy")
        tgt_vscore_path = os.path.join(getattr(ViTExperiment, ds_name).OUTPUT_DIR.format(k=k), f"misclf_top{tgt_rank}", vname, f"vscore_l1tol{end_li}_all_label_ori_{tgt_split}_cor.npy")
        os.system(f"cp {cor_vscore_path} {tgt_vscore_path}")
    all_bhs = []
    all_ahs = []
    all_mhs = []
    all_logits = []
    # loop for dataset batch
    tgt_ds = ds_preprocessed[tgt_split].select(indices=tgt_mis_indices)
    for entry_dic in tqdm(tgt_ds.iter(batch_size=batch_size), total=len(tgt_ds)//batch_size+1):
        x, y = entry_dic["pixel_values"].to(device), entry_dic["labels"]
        output = model.forward(x, tgt_pos=tgt_pos, output_hidden_states_before_layernorm=False, output_intermediate_states=True)
        # CLSトークンに対応するintermediate statesを取得
        num_layer = len(output.intermediate_states)
        bhs, ahs, mhs = [], [], []
        for i in range(num_layer):
            # output.intermediate_states[i]はiレイヤめの(中間ニューロンの前のニューロン, 中間ニューロン，中間ニューロンの後のニューロン)がタプルで入っている
            bhs.append(output.intermediate_states[i][0])
            mhs.append(output.intermediate_states[i][1])
            ahs.append(output.intermediate_states[i][2]) # ここの時点ではレイヤの次元が先に来る
        bhs = np.array(bhs).transpose(1, 0, 2) # (batch_size, num_layers, num_neurons)
        mhs = np.array(mhs).transpose(1, 0, 2) # (batch_size, num_layers, num_neurons)
        ahs = np.array(ahs).transpose(1, 0, 2) # (batch_size, num_layers, num_neurons)
        all_bhs.append(bhs)
        all_mhs.append(mhs)
        all_ahs.append(ahs)
        logits = output.logits.cpu().detach().numpy()
        all_logits.append(logits)
    all_logits = np.concatenate(all_logits) # (num_samples, num_labels)
    all_pred_labels = np.argmax(all_logits, axis=-1) # (num_samples, )
    if misclf_type == "src_tgt":
        # all_pred_labelsがすべてslabelになっていることを確認
        assert all(all_pred_labels == slabel), f"all_pred_labels: {all_pred_labels}"
    elif misclf_type == "tgt":
        for pred_l, true_l in zip(all_pred_labels, labels[tgt_split][tgt_mis_indices]):
            # pred_lとtrue_lが異なることを保証
            assert pred_l != true_l, f"pred_l: {pred_l}, true_l: {true_l}"
            # # pred_lかtrue_lがtlabelであることを保証
            # assert pred_l == tlabel or true_l == tlabel, f"pred_l: {pred_l}, true_l: {true_l}"
    all_bhs = np.concatenate(all_bhs)
    all_ahs = np.concatenate(all_ahs)
    all_mhs = np.concatenate(all_mhs)
    print(f"all_bhs: {all_bhs.shape}, all_ahs: {all_ahs.shape}, all_mhs: {all_mhs.shape}, all_logits: {all_logits.shape}")

    # 正解/不正解データに対するv-scoreの計算を行い，保存する
    for all_states, vscore_dir in zip([all_bhs, all_ahs, all_mhs], [vscore_before_dir, vscore_after_dir, vscore_med_dir]): # bhs, ahs での繰り返し
        # 対象のレイヤに対してvscoreを計算
        # =================================
        vscore_per_layer = []
        for tgt_layer in range(end_li):
            print(f"tgt_layer: {tgt_layer}")
            tgt_mid_state = all_states[:, tgt_layer, :] # (num_samples, num_neurons)
            # vscoreを計算
            vscore = get_vscore(tgt_mid_state)
            vscore_per_layer.append(vscore)
        vscores = np.array(vscore_per_layer) # (num_tgt_layer, num_neurons)
        # vscoresを保存
        ds_type = f"ori_{tgt_split}" if fpfn is None else f"ori_{tgt_split}_{fpfn}"
        if misclf_type == "src_tgt":
            vscore_save_path = os.path.join(vscore_dir, f"vscore_l1tol{end_li}_{slabel}to{tlabel}_{ds_type}_mis.npy")
        elif misclf_type == "tgt":
            vscore_save_path = os.path.join(vscore_dir, f"vscore_l1tol{end_li}_{tlabel}_{ds_type}_mis.npy")
        np.save(vscore_save_path, vscores)
        print(f"vscore ({vscores.shape}) saved at {vscore_save_path}") # mid_statesがnan (correct or incorrect predictions の数が 0) の場合はvscoreもnanになる
    
    # NOTE: 中間ニューロン状態自体の保存は必要そうになったら実装する
    # # 各サンプルに対するレイヤごとの隠れ状態を保存していく
    # # 将来的なことを考えてnumpy->tensorに変換してから保存
    # num_layers = model.vit.config.num_hidden_layers
    # for l_idx in range(num_layers):
    #     # 特定のレイヤのstatesだけ抜き出し
    #     tgt_mid_states = torch.tensor(all_mid_states[:, l_idx, :]).cpu()
    #     # 保存
    #     intermediate_save_path = os.path.join(cache_dir, f"intermediate_states_l{l_idx}.pt")
    #     torch.save(tgt_mid_states, intermediate_save_path)
    #     print(f"tgt_mid_states: {tgt_mid_states.shape} is saved at {intermediate_save_path}")