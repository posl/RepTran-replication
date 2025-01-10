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

NUM_IDENTIFIED_NEURONS = Experiment1.NUM_IDENTIFIED_NEURONS # exp-fl-1.md参照



def main(ds_name, k, tgt_rank, misclf_type, fpfn):
    ts = time.perf_counter()
    
    # デバイス (cuda, or cpu) の取得
    device = get_device()
    # datasetごとに違う変数のセット
    tgt_split = "repair" # NOTE: we only use repair split for repairing
    tgt_layer = 11 # NOTE: we only use the last layer for repairing
    tgt_pos = ViTExperiment.CLS_IDX
    num_points = ViTExperiment.NUM_POINTS # integrated gradientの積分近似の分割数
    
    # datasetをロード (true_labelsが欲しいので)
    ds_dirname = f"{ds_name}_fold{k}"
    ds = load_from_disk(os.path.join(ViTExperiment.DATASET_DIR, ds_dirname))
    label_col = "fine_label"
    # ラベルの取得 (shuffleされない)
    labels = {
        "train": np.array(ds["train"][label_col]),
        "repair": np.array(ds["repair"][label_col]),
        "test": np.array(ds["test"][label_col])
    }
    true_labels = labels[tgt_split]
    
    # pretrained modelのロード
    pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR.format(k=k)
    # location informationの保存先
    save_dir = os.path.join(pretrained_dir, f"all_weights_location")
    model = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
    model.eval()
    vit_from_last_layer = ViTFromLastLayer(model)
    vit_from_last_layer.eval()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # tgt_rankの誤分類情報を取り出す
    misclf_info_dir = os.path.join(pretrained_dir, "misclf_info")
    misclf_pair, tgt_label, tgt_mis_indices = identfy_tgt_misclf(misclf_info_dir, tgt_split=tgt_split, tgt_rank=tgt_rank, misclf_type=misclf_type, fpfn=fpfn)
    indices_to_incorrect = tgt_mis_indices
    
    # original model の repair setの各サンプルに対する正解/不正解のインデックスを取得
    pred_res_dir = os.path.join(pretrained_dir, "pred_results", "PredictionOutput")
    if misclf_type == "tgt":
        ori_pred_labels, is_correct, indices_to_correct, is_correct_others, indices_to_correct_others = get_ori_model_predictions(pred_res_dir, labels, tgt_split=tgt_split, misclf_type=misclf_type, tgt_label=tgt_label)
    else:
        ori_pred_labels, is_correct, indices_to_correct = get_ori_model_predictions(pred_res_dir, labels, tgt_split=tgt_split, misclf_type=misclf_type, tgt_label=tgt_label)
    print(f"len(indices_to_correct): {len(indices_to_correct)}, len(indices_to_incorrect): {len(indices_to_incorrect)}")
    
    # 正解データからrepairに使う一定数だけランダムに取り出す
    sampled_indices_to_correct = sample_from_correct_samples(len(indices_to_incorrect), indices_to_correct)
    # 抽出した正解データと，全不正解データを結合して1つのデータセットにする
    tgt_indices = sampled_indices_to_correct.tolist() + indices_to_incorrect.tolist() # .tolist() は 非破壊的method
    # tgt_indicesは全てユニークな値であることを保証
    assert len(tgt_indices) == len(set(tgt_indices)), f"len(tgt_indices): {len(tgt_indices)}, len(set(tgt_indices)): {len(set(tgt_indices))}"
    print(f"len(tgt_indices): {len(tgt_indices)})")
    # tgt_indicesに対応するデータトラベルを取り出す
    tgt_ds = ds[tgt_split].select(tgt_indices)
    tgt_labels = labels[tgt_split][tgt_indices]
    # FLに使う各サンプルの予測ラベルと正解ラベルを表示
    print(f"ori_pred_labels[tgt_indices]: {ori_pred_labels[tgt_indices]} (len: {len(ori_pred_labels[tgt_indices])})")
    print(f"ori_tgt_labels[tgt_indices]: {tgt_labels} (len: {len(tgt_labels)})")
    
    # キャッシュの保存用のディレクトリ
    cache_dir = os.path.join(pretrained_dir, f"cache_hidden_states_before_layernorm_{tgt_split}")
    cache_path = os.path.join(cache_dir, f"hidden_states_before_layernorm_{tgt_layer}.npy")
    # cache_pathに存在することを確認
    assert os.path.exists(cache_path), f"cache_path: {cache_path} does not exist."
    cached_hidden_states = np.load(cache_path)
    hidden_states_before_layernorm = torch.from_numpy(cached_hidden_states).to(device)
    # 利用するインデックスのデータだけ取り出す
    hidden_states_before_layernorm = hidden_states_before_layernorm[tgt_indices]
    print(f"hidden_states_before_layernorm.shape: {hidden_states_before_layernorm.shape}")
    assert len(hidden_states_before_layernorm) == len(tgt_indices), f"len(hidden_states_before_layernorm): {len(hidden_states_before_layernorm)}, len(tgt_indices): {len(tgt_indices)}"
    batch_size = ViTExperiment.BATCH_SIZE
    num_batches = (hidden_states_before_layernorm.shape[0] + batch_size - 1) // batch_size  # バッチの数を計算 (最後の中途半端なバッチも使いたいので，切り上げ)
    batched_hidden_states_before_layernorm = np.array_split(hidden_states_before_layernorm, num_batches) # hidden_states_before_layernormをnum_batches個のsubarrayに分ける
    # tgt_labelsも同様にバッチにする
    batched_tgt_labels = np.array_split(tgt_labels, num_batches)
    
    # ViTFromLastLayerで予測をして最終層の中間ニューロンに対するロスの勾配を取得
    grad_dict = defaultdict(dict)
    for cached_state, tls in tqdm(
        zip(batched_hidden_states_before_layernorm, batched_tgt_labels),
        total=len(batched_hidden_states_before_layernorm),
    ):
        optimizer.zero_grad()  # サンプルごとに勾配を初期化
        # ここでViTFromLastLayerのforwardが実行される
        logits = vit_from_last_layer(hidden_states_before_layernorm=cached_state,tgt_pos=tgt_pos)
        loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = loss_fn(logits, torch.tensor(tls).to(device)) # バッチ内のサンプルに対するロスの勾配の平均
        loss.backward(retain_graph=True)  # 勾配計算 (retain_graph=Trueで計算グラフを保持)
        # NOTE: 各サンプルに対するロスの平均の勾配 = 各サンプルに対する勾配の平均
        
        # ForwardImpact計算用データ（重みの入力となるニューロンが必要）
        cached_state_aft_ln = vit_from_last_layer.base_model_last_layer.layernorm_after(cached_state)
        cached_state_aft_mid = vit_from_last_layer.base_model_last_layer.intermediate(cached_state_aft_ln)
        # cached_stateの2軸目がtgt_posの部分だけ取り出す
        cached_state_aft_ln = cached_state_aft_ln[:, tgt_pos, :]
        cached_state_aft_mid = cached_state_aft_mid[:, tgt_pos, :]
        # print(f"cached_state_aft_ln.shape: {cached_state_aft_ln.shape}") # shape: (batch_size, in_dim)
        # print(f"cached_state_aft_mid.shape: {cached_state_aft_mid.shape}") # shape: (batch_size, mid_dim)
        
        # TODO: 関数化しないときつい
        # バッチ内のサンプル全体の平均としてBIとFIを計算してリストに保存する．
        # んで，全バッチ終わったらそれらを平均する．
        
        for ba, cs, tgt_component in zip(
            ["before", "after"],
            [cached_state_aft_ln, cached_state_aft_mid],
            [vit_from_last_layer.base_model_last_layer.intermediate.dense, vit_from_last_layer.base_model_last_layer.output.dense]
        ):
            grad_dict[ba] = defaultdict(list)
            
            # ロスの勾配のバッチ内の平均 (BackwardImpact)
            # ===============================================
            # print(tgt_component.weight.grad.cpu().shape) # 重みと同じ形状 (out_dim, in_dim)
            grad_dict[ba]["bw"].append(tgt_component.weight.grad.cpu())
            
            # logitsの勾配 * (ニューロン*重み) のバッチ内の平均 (ForwardImpact)
            # ===============================================
            # logitsの勾配
            grad_out_weight = torch.autograd.grad(
                logits, tgt_component.weight, grad_outputs=torch.ones_like(logits), retain_graph=True
            )[0] # 重みと同じ形状 (out_dim, in_dim)
            # (ニューロン*重み)
            # tgt_weight を (1, out_dim, in_dim) に変形
            tgt_weight_expanded = tgt_component.weight.unsqueeze(0)  # バッチ方向に次元追加
            # oi を (bsize, 1, in_din) に変形
            oi_expanded = cs.unsqueeze(1)  # 行方向に次元追加
            impact_out_weight = tgt_weight_expanded * oi_expanded  # shape: (bsize, out_dim, in_dim)
            normalization_terms = impact_out_weight.sum(dim=2) # shape: (bsize, out_dim)
            normalized_impact_out_weight = impact_out_weight / (normalization_terms[:,:, None] + 1e-8) # shape: (bsize, out_dim, in_dim)
            # バッチ軸で平均
            mean_normalized_impact_out_weight = normalized_impact_out_weight.mean(dim=0) # shape: (out_dim, in_dim)
            assert mean_normalized_impact_out_weight.shape == grad_out_weight.shape, f"mean_normalized_impact_out_weight.shape: {mean_normalized_impact_out_weight.shape}, grad_out_weight.shape: {grad_out_weight.shape}"
            grad_dict[ba]["fw"].append(mean_normalized_impact_out_weight * grad_out_weight)

    for ba in ["before", "after"]:
        # grad_dict[ba]のそれぞれの要素を平均
        grad_bw = torch.mean(torch.stack(grad_dict[ba]["bw"]), dim=0)
        grad_fw = torch.mean(torch.stack(grad_dict[ba]["fw"]), dim=0)
        print(f"grad_bw.shape: {grad_bw.shape}, grad_fw.shape: {grad_fw.shape}")
    
    # ここまでで，中間レイヤのbefore, afterそれぞれの各重みに対して bw, fw の値が得られている．
    # あとは，これらのスコアに順位をつけて，上位n個の重みの位置情報を保存する．
    places_to_fix = []
    
    
    n = NUM_IDENTIFIED_NEURONS
    
    # 最終的に，location_save_pathに各中間ニューロンの重みの位置情報を保存する
    if fpfn is not None and misclf_type == "tgt":
        location_save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_{fpfn}_weights_location")
    elif misclf_type == "all":
        location_save_dir = os.path.join(pretrained_dir, f"all_weights_location")
    else:
        location_save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_weights_location")
    location_save_path = os.path.join(location_save_dir, f"location_n{n}_neuron_bl.npy")
    np.save(location_save_path, places_to_fix)
    print(f"saved location information to {location_save_path}")
    # 終了時刻
    te = time.perf_counter()
    elapsed_time = te - ts
    return elapsed_time
    
if __name__ == "__main__":
    ds = "c100"
    k_list = range(5)
    tgt_rank_list = range(1, 6)
    misclf_type_list = ["all", "src_tgt", "tgt"]
    fpfn_list = [None, "fp", "fn"]
    for k, tgt_rank, misclf_type, fpfn in product(k_list, tgt_rank_list, misclf_type_list, fpfn_list):
        print(f"Start: ds={ds}, k={k}, tgt_rank={tgt_rank}, misclf_type={misclf_type}, fpfn={fpfn}")
        if (misclf_type == "src_tgt" or misclf_type == "all") and fpfn is not None: # misclf_type == "src_tgt" or "all"の時はfpfnはNoneだけでいい
            continue
        if misclf_type == "all" and tgt_rank != 1: # misclf_type == "all"の時にtgt_rankは関係ないのでこのループもスキップすべき
            continue
        main(ds, k, tgt_rank, misclf_type, fpfn)
    