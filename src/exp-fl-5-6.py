import os, sys, time, pickle, json, math
from tqdm import tqdm
from collections import defaultdict
from itertools import product
import numpy as np
import pandas as pd
from utils.helper import get_device, json2dict
from utils.vit_util import ViTFromLastLayer, identfy_tgt_misclf, get_ori_model_predictions, get_vscore, localize_neurons_with_mean_activation, get_batched_hs, get_batched_labels
from utils.constant import ViTExperiment, Experiment1
from utils.log import set_exp_logging
from logging import getLogger
from datasets import load_from_disk
from transformers import ViTForImageClassification
import torch
import torch.optim as optim

NUM_IDENTIFIED_NEURONS = Experiment1.NUM_IDENTIFIED_NEURONS

# デバイス (cuda, or cpu) の取得
device = get_device()

def calculate_vdfif_mean_act_grad_loss(batched_hidden_states, batched_labels, vit_from_last_layer, optimizer, tgt_pos):
    
    for cached_state, tls in tqdm(
        zip(batched_hidden_states, batched_labels),
        total=len(batched_hidden_states),
    ):
        optimizer.zero_grad()  # サンプルごとに勾配を初期化

        # Forward pass
        logits = vit_from_last_layer(hidden_states_before_layernorm=cached_state, tgt_pos=tgt_pos)
        loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = loss_fn(logits, torch.tensor(tls).to(device))  # バッチ内のサンプルに対するロスの平均
        loss.backward(retain_graph=True)

def main(ds_name, k, tgt_rank, misclf_type, fpfn, n, sample_from_correct=False, strategy="weighted"):
    sample_from_correct = True
    ts = time.perf_counter()
    
    # datasetごとに違う変数のセット
    tgt_split = "repair" # NOTE: we only use repair split for repairing
    tgt_layer = 11 # NOTE: we only use the last layer for repairing
    tgt_pos = ViTExperiment.CLS_IDX
    
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
    
    # exp-fl-5の結果保存用ディレクトリ
    exp_dir = os.path.join("./exp-fl-5", f"{ds_name}_fold{k}")
    
    # tgt_rankの誤分類情報を取り出す
    misclf_info_dir = os.path.join(exp_dir, "misclf_info")
    misclf_pair, tgt_label, tgt_mis_indices = identfy_tgt_misclf(misclf_info_dir, tgt_split=tgt_split, tgt_rank=tgt_rank, misclf_type=misclf_type, fpfn=fpfn)
    indices_to_incorrect = tgt_mis_indices
    if misclf_type == "src_tgt":
        slabel, tlabel = misclf_pair
    elif misclf_type == "tgt":
        tlabel = tgt_label
    
    # original model の repair setの各サンプルに対する正解/不正解のインデックスを取得
    pred_res_dir = os.path.join(exp_dir, "PredictionOutput")
    if misclf_type == "tgt":
        ori_pred_labels, is_correct, indices_to_correct, is_correct_others, indices_to_correct_others = get_ori_model_predictions(pred_res_dir, labels, tgt_split=tgt_split, misclf_type=misclf_type, tgt_label=tgt_label)
    else:
        ori_pred_labels, is_correct, indices_to_correct = get_ori_model_predictions(pred_res_dir, labels, tgt_split=tgt_split, misclf_type=misclf_type, tgt_label=tgt_label)
    print(f"len(indices_to_correct): {len(indices_to_correct)}, len(indices_to_incorrect): {len(indices_to_incorrect)}")

    # 抽出した正解データと，全不正解データを結合して1つのデータセットにする
    tgt_indices = indices_to_correct.tolist() + indices_to_incorrect.tolist() # .tolist() は 非破壊的method
    # tgt_indicesは全てユニークな値であることを保証
    assert len(tgt_indices) == len(set(tgt_indices)), f"len(tgt_indices): {len(tgt_indices)}, len(set(tgt_indices)): {len(set(tgt_indices))}"
    print(f"len(tgt_indices): {len(tgt_indices)})")
    # tgt_indicesに対応するデータトラベルを取り出す
    tgt_labels = labels[tgt_split][tgt_indices]
    # FLに使う各サンプルの予測ラベルと正解ラベルを表示
    print(f"ori_pred_labels[tgt_indices]: {ori_pred_labels[tgt_indices]} (len: {len(ori_pred_labels[tgt_indices])})")
    print(f"ori_tgt_labels[tgt_indices]: {tgt_labels} (len: {len(tgt_labels)})")
    
    # キャッシュの保存用のディレクトリ
    # hs_bef_norm_dir
    hs_bef_norm_dir = os.path.join(exp_dir, f"cache_hidden_states_before_layernorm_{tgt_split}")
    hs_bef_norm_path = os.path.join(hs_bef_norm_dir, f"hidden_states_before_layernorm_{tgt_layer}.npy")
    # intermediate_state
    mid_dir = os.path.join(exp_dir, f"cache_states_{tgt_split}")
    mid_path = os.path.join(mid_dir, f"intermediate_states_l{tgt_layer}.npy")
    # cache_pathに存在することを確認
    assert os.path.exists(hs_bef_norm_path), f"cache_path: {hs_bef_norm_path} does not exist."
    assert os.path.exists(mid_path), f"cache_path: {mid_path} does not exist."
    # vit_utilsの関数を使ってバッチを取得
    batch_size = ViTExperiment.BATCH_SIZE
    
    # Vscore保存用ディレクトリ
    if misclf_type == "all":
        vscore_dir = os.path.join(exp_dir, "vscores")
    else:
        vscore_dir = os.path.join(exp_dir, f"misclf_top{tgt_rank}", "vscores")
    os.makedirs(vscore_dir, exist_ok=True)
    # mid_pathをロードしてvscoreを取得
    intermediate_states = np.load(mid_path)
    
    # 成功時と失敗時のvscoreを分けて保存
    for cor_mis, indices in zip(["cor", "mis"], [indices_to_correct, indices_to_incorrect]):
        # Vscoreの保存ファイル名
        ds_type = f"ori_{tgt_split}" if fpfn is None else f"ori_{tgt_split}_{fpfn}"
        if misclf_type == "src_tgt":
            vscore_save_path = os.path.join(vscore_dir, f"vscore_l1tol{tgt_layer+1}_{slabel}to{tlabel}_{ds_type}_{cor_mis}.npy")
        elif misclf_type == "tgt":
            vscore_save_path = os.path.join(vscore_dir, f"vscore_l1tol{tgt_layer+1}_{tlabel}_{ds_type}_{cor_mis}.npy")
        else:
            assert misclf_type == "all", f"misclf_type: {misclf_type} is not supported."
            vscore_save_path = os.path.join(vscore_dir, f"vscore_l1tol{tgt_layer+1}_all_label_{ds_type}_{cor_mis}.npy")
        # cor_misに対応するデータに対する中間状態の取り出し
        tgt_mid_states = intermediate_states[indices]
        # Vscoreの計算
        tgt_vscore = get_vscore(tgt_mid_states)
        np.save(vscore_save_path, tgt_vscore)
        # misclf_type == "all" かつ "cor_mis" == "cor" の時は追加で各topnのディレクトリにも保存したい
        if misclf_type == "all" and cor_mis == "cor":
            for tr in range(1, 6):
                additional_vscore_save_dir = os.path.join(exp_dir, f"misclf_top{tr}", "vscores")
                if not os.path.exists(additional_vscore_save_dir):
                    os.makedirs(additional_vscore_save_dir)
                additional_vscore_save_path = os.path.join(additional_vscore_save_dir, f"vscore_l1tol{tgt_layer+1}_all_label_{ds_type}_{cor_mis}.npy")
                np.save(additional_vscore_save_path, tgt_vscore)
        print(f"{cor_mis} vscore has been saved at {vscore_save_path}. shape is {tgt_vscore.shape}")

    # localize_neurons_with_mean_activationによるニューロン特定 (ニューロン数は限定しないので (layer_idx, neuron_idx) のリストがスコアの降順で帰ってくる) 
    places_to_neuron, tgt_neuron_score = localize_neurons_with_mean_activation(vscore_before_dir=None, vscore_dir=vscore_dir, vscore_after_dir=None, tgt_layer=tgt_layer, n=None, intermediate_states=intermediate_states, tgt_mis_indices=tgt_mis_indices, misclf_pair=misclf_pair, tgt_label=tgt_label, fpfn=fpfn)
    
    # ニューロン位置情報を保存
    if fpfn is not None and misclf_type == "tgt":
        location_save_dir = os.path.join(exp_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_{fpfn}_weights_location")
    elif misclf_type == "all":
        location_save_dir = os.path.join(exp_dir, f"all_weights_location")
    else:
        location_save_dir = os.path.join(exp_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_weights_location")
    if not os.path.exists(location_save_dir):
        os.makedirs(location_save_dir)
    location_save_path = os.path.join(location_save_dir, f"exp-fl-5_location_nAll_wAll_neuron.npy")
    np.save(location_save_path, places_to_neuron)
    print(f"saved location information to {location_save_path}")
    print(f"len(places_to_neuron): {len(places_to_neuron)}")
    print(f"tgt_neuron_score.shape: {tgt_neuron_score.shape}")
    print(f"tgt_neuron_score: {tgt_neuron_score}")
    
    
    # ここまでで Vdiff x Use_i によるニューロン特定ができたので，次は勾配も使った重み特定をする．
    
    # 正解サンプル (I_pos) と誤りサンプル (I_neg) を分割
    # correct_batched_hidden_states = get_batched_hs(cache_path, batch_size)
    # correct_batched_labels = get_batched_labels(labels[tgt_split], batch_size)
    incorrect_batched_hidden_states = get_batched_hs(hs_bef_norm_path, batch_size, tgt_mis_indices)
    incorrect_batched_labels = get_batched_labels(labels[tgt_split], batch_size, tgt_mis_indices)
    
    # ロスの勾配の取得に必要なモデルをロード
    pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR.format(k=k)
    model = ViTForImageClassification.from_pretrained(os.path.join(pretrained_dir, "checkpoint-1250")).to(device)
    model.eval()
    vit_from_last_layer = ViTFromLastLayer(model)
    vit_from_last_layer.eval()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # 結果の保存用
    results = defaultdict(lambda: torch.tensor([])) # key: bef or aft, value: mean of grad_loss of the weights (shape: (out_dim, in_dim))
    
    # 誤分類サンプル集合に対してWbef, Waftの重みのロスに対する勾配を取得
    for cached_state, tls in tqdm(
        zip(incorrect_batched_hidden_states, incorrect_batched_labels),
        total=len(incorrect_batched_hidden_states),
    ):
        optimizer.zero_grad()  # サンプルごとに勾配を初期化

        # Forward pass
        logits = vit_from_last_layer(hidden_states_before_layernorm=cached_state, tgt_pos=tgt_pos)
        loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = loss_fn(logits, torch.tensor(tls).to(device))  # バッチ内のサンプルに対するロスの平均
        loss.backward(retain_graph=True)
        
        for tgt_component, ba_layer in zip(
            [vit_from_last_layer.base_model_last_layer.intermediate.dense, vit_from_last_layer.base_model_last_layer.output.dense],
            ["before", "after"],
        ):
            # BI: ロスの勾配
            grad = tgt_component.weight.grad.cpu()
            # print(f"{ba_layer} - grad.shape: {grad.shape}")  # shape: (out_dim, in_dim)
            if len(results[ba_layer]) == 0:
                results[ba_layer] = grad.detach().clone().cpu()
            else:
                results[ba_layer] += grad.detach().cpu()
    
    # バッチ全体の平均を計算
    for ba_layer in ["before", "after"]:
        results[ba_layer] /= len(incorrect_batched_hidden_states)
        print(f"{ba_layer} - results[ba_layer].shape: {results[ba_layer].shape}")
    
    # ニューロンの特定結果と勾配の結果を踏まえて，重みの特定
    indices_to_neurons = [x[1] for x in places_to_neuron] # places_to_neuronは [レイヤ番号, ニューロン番号]のリストになっているが以降の操作では最終層なのは自明なのでニューロン番号だけを取り出す
    
    # grad_Wbef[idx, :] (行ベクトル) と grad_Waft[:, idx] (列ベクトル) を取り出し
    grad_Wbef_row = results["before"][indices_to_neurons, :]  # shape: (num_tgt_neuron, 768)
    grad_Waft_col = results["after"][:, indices_to_neurons]  # shape: (768, num_tgt_neuron)
    print(f"grad_Wbef_row.shape: {grad_Wbef_row.shape}, grad_Waft_col.shape: {grad_Waft_col.shape}")
    
    # フラット化
    grad_Wbef_row_flat = grad_Wbef_row.flatten()  # shape: (num_neurons * 768,)
    grad_Waft_col_flat = grad_Waft_col.flatten()  # shape: (768 * num_neurons,)

    # 結合
    combined = np.concatenate((grad_Wbef_row_flat, grad_Waft_col_flat))  # shape: (num_neurons * 768 + 768 * num_neurons,)
    
    # 上位X個の値とインデックスを取得
    top_indices = np.argsort(-np.abs(combined))  # 絶対値が大きい順でソート
    top_values = combined[top_indices]
    print(f"len(top_indices): {len(top_indices)}") # w_num
    # インデックスを元の形状に変換
    original_shape_bef = grad_Wbef_row.shape  # (num_neurons, 768)
    original_shape_aft = grad_Waft_col.shape  # (768, num_neurons)
    
    top_indices_bef = [i for i in top_indices if i < len(grad_Wbef_row_flat)]  # Wbef に対応するインデックス
    top_indices_aft = [i - len(grad_Wbef_row_flat) for i in top_indices if i >= len(grad_Wbef_row_flat)]  # Waft に対応
    assert len(top_indices_bef) + len(top_indices_aft) == len(top_indices)

    # unravel で元の形状に戻す
    pos_before = np.array([np.unravel_index(idx, original_shape_bef) for idx in top_indices_bef])
    pos_after = np.array([np.unravel_index(idx, original_shape_aft) for idx in top_indices_aft])

    # 位置情報を保存
    location_save_path = os.path.join(location_save_dir, f"exp-fl-5_location_nAll_wAll_weight.npy")
    np.save(location_save_path, (pos_before, pos_after))
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
    results = []
    n_list = [96]
    for k, tgt_rank, misclf_type, fpfn, n in product(k_list, tgt_rank_list, misclf_type_list, fpfn_list, n_list):
        print(f"Start: ds={ds}, k={k}, tgt_rank={tgt_rank}, misclf_type={misclf_type}, fpfn={fpfn}")
        if (misclf_type == "src_tgt" or misclf_type == "all") and fpfn is not None: # misclf_type == "src_tgt" or "all"の時はfpfnはNoneだけでいい
            continue
        if misclf_type == "all" and tgt_rank != 1: # misclf_type == "all"の時にtgt_rankは関係ないのでこのループもスキップすべき
            continue
        elapsed_time = main(ds, k, tgt_rank, misclf_type, fpfn, n=NUM_IDENTIFIED_NEURONS)
        results.append({"ds": ds, "k": k, "tgt_rank": tgt_rank, "misclf_type": misclf_type, "fpfn": fpfn, "elapsed_time": elapsed_time})
    # results を csv にして保存
    result_df = pd.DataFrame(results)
    result_df.to_csv("./exp-fl-5-6_time.csv", index=False)