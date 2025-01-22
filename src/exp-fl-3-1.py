import os, sys, time, pickle, json, math
from tqdm import tqdm
from collections import defaultdict
from itertools import product
import numpy as np
import pandas as pd
import torch
from utils.helper import get_device, json2dict
from utils.vit_util import identfy_tgt_misclf, localize_neurons_with_mean_activation, get_batched_hs, get_batched_labels, ViTFromLastLayer
from utils.constant import ViTExperiment, Experiment1, Experiment3
from utils.log import set_exp_logging
from logging import getLogger
from datasets import load_from_disk
from transformers import ViTForImageClassification
import torch
import torch.optim as optim

logger = getLogger("base_logger")
device = get_device()

def main(ds_name, k, tgt_rank, misclf_type, fpfn):
    print(f"ds_name: {ds_name}, fold_id: {k}, tgt_rank: {tgt_rank}, misclf_type: {misclf_type}, fpfn: {fpfn}")
    
    
    ts = time.perf_counter()
    # 変更する対象を決定
    n_ratio = Experiment3.NUM_IDENTIFIED_NEURONS_RATIO # exp-fl-3.md参照
    w_num = Experiment3.NUM_TOTAL_WEIGHTS
    
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
    tgt_pos = ViTExperiment.CLS_IDX
    
    # 結果とかログの保存先を先に作っておく
    # pretrained modelのディレクトリ
    pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR.format(k=k)
    save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_weights_location")
    if misclf_type == "all":
        save_dir = os.path.join(pretrained_dir, f"all_weights_location")
    if fpfn is not None and misclf_type == "tgt":
        save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_{fpfn}_weights_location")
    os.makedirs(save_dir, exist_ok=True)
        
    # このpythonのファイル名を取得
    this_file_name = os.path.basename(__file__).split(".")[0]
    exp_name = f"exp-fl-3_{this_file_name}_n{n_ratio}"
    # loggerの設定をして設定情報を表示
    logger = set_exp_logging(exp_dir=save_dir, exp_name=exp_name)
    logger.info(f"ds_name: {ds_name}, fold_id: {k}, tgt_rank: {tgt_rank}, n_ratio: {n_ratio}, w_num: {w_num}, misclf_type: {misclf_type}")

    # tgt_rankの誤分類情報を取り出す
    tgt_split = "repair" # NOTE: we only use repair split for repairing
    tgt_layer = 11 # NOTE: we only use the last layer for repairing
    logger.info(f"tgt_layer: {tgt_layer}, tgt_split: {tgt_split}")
    misclf_info_dir = os.path.join(pretrained_dir, "misclf_info")
    misclf_pair, tgt_label, tgt_mis_indices = identfy_tgt_misclf(misclf_info_dir, tgt_split=tgt_split, tgt_rank=tgt_rank, misclf_type=misclf_type, fpfn=fpfn)
    logger.info(f"misclf_pair: {misclf_pair}, tgt_label: {tgt_label}, len(tgt_mis_indices): {len(tgt_mis_indices)}")
    
    # 中間ニューロン値のキャッシュのロード
    mid_cache_dir = os.path.join(pretrained_dir, f"cache_states_{tgt_split}")
    mid_save_path = os.path.join(mid_cache_dir, f"intermediate_states_l{tgt_layer}.pt")
    cached_mid_states = torch.load(mid_save_path, map_location="cpu") # (tgt_splitのサンプル数(10000), 中間ニューロン数(3072))
    # cached_mid_statesをnumpy配列にする
    cached_mid_states = cached_mid_states.detach().numpy().copy()
    print(f"cached_mid_states.shape: {cached_mid_states.shape}")

    # ===============================================
    # localization phase
    # ===============================================

    if misclf_type == "src_tgt" or misclf_type == "tgt":
        vscore_before_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", "vscores_before")
        vscore_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", "vscores")
        vscore_after_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", "vscores_after")
    elif misclf_type == "all":
        vscore_before_dir = os.path.join(pretrained_dir, "vscores_before")
        vscore_dir = os.path.join(pretrained_dir, "vscores")
        vscore_after_dir = os.path.join(pretrained_dir, "vscores_after")
    logger.info(f"vscore_before_dir: {vscore_before_dir}")
    logger.info(f"vscore_dir: {vscore_dir}")
    logger.info(f"vscore_after_dir: {vscore_after_dir}")
    # vscoreとmean_activationを用いたlocalizationを実行
    method_name = "vscore_mean_activation"
    location_save_path = os.path.join(save_dir, f"exp-fl-3_location_n{n_ratio}_w{w_num}_neuron.npy")
    places_to_neuron, tgt_neuron_score = localize_neurons_with_mean_activation(vscore_before_dir, vscore_dir, vscore_after_dir, tgt_layer, n_ratio, intermediate_states=cached_mid_states, tgt_mis_indices=tgt_mis_indices, misclf_pair=misclf_pair, tgt_label=tgt_label, fpfn=fpfn)
    # log表示
    logger.info(f"places_to_neuron={places_to_neuron}")
    logger.info(f"num(pos_to_fix)={len(places_to_neuron)}")
    # 位置情報を保存
    np.save(location_save_path, places_to_neuron)
    logger.info(f"saved location information to {location_save_path}")
    print(f"saved location information to {location_save_path}")
    print(f"len(places_to_neuron): {len(places_to_neuron)}")
    print(f"tgt_neuron_score.shape: {tgt_neuron_score.shape}")
    print(f"tgt_neuron_score: {tgt_neuron_score}")
    
    # ここまでで Vdiff x Use_i によるニューロン特定ができたので，次は勾配も使った重み特定をする．
    
    # キャッシュの保存用のディレクトリ
    cache_dir = os.path.join(pretrained_dir, f"cache_hidden_states_before_layernorm_{tgt_split}")
    cache_path = os.path.join(cache_dir, f"hidden_states_before_layernorm_{tgt_layer}.npy")
    # cache_pathに存在することを確認
    assert os.path.exists(cache_path), f"cache_path: {cache_path} does not exist."
    # vit_utilsの関数を使ってバッチを取得
    batch_size = ViTExperiment.BATCH_SIZE
    
    # 正解サンプル (I_pos) と誤りサンプル (I_neg) を分割
    # correct_batched_hidden_states = get_batched_hs(cache_path, batch_size)
    # correct_batched_labels = get_batched_labels(labels[tgt_split], batch_size)
    incorrect_batched_hidden_states = get_batched_hs(cache_path, batch_size, tgt_mis_indices)
    incorrect_batched_labels = get_batched_labels(labels[tgt_split], batch_size, tgt_mis_indices)
    
    # ロスの勾配の取得に必要なモデルをロード
    model = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
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
    top_indices = np.argsort(-np.abs(combined))[:w_num]  # 絶対値が大きい順でソート
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
    location_save_path = os.path.join(save_dir, f"exp-fl-3_location_n{n_ratio}_w{w_num}_weight.npy")
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
    
    for k, tgt_rank, misclf_type, fpfn in product(k_list, tgt_rank_list, misclf_type_list, fpfn_list):
        print(f"Start: ds={ds}, k={k}, tgt_rank={tgt_rank}, misclf_type={misclf_type}, fpfn={fpfn}")
        if (misclf_type == "src_tgt" or misclf_type == "all") and fpfn is not None: # misclf_type == "src_tgt" or "all"の時はfpfnはNoneだけでいい
            continue
        if misclf_type == "all" and tgt_rank != 1: # misclf_type == "all"の時にtgt_rankは関係ないのでこのループもスキップすべき
            continue
        elapsed_time = main(ds, k, tgt_rank, misclf_type, fpfn)
        results.append({"ds": ds, "k": k, "tgt_rank": tgt_rank, "misclf_type": misclf_type, "fpfn": fpfn, "elapsed_time": elapsed_time})
    # results を csv にして保存
    result_df = pd.DataFrame(results)
    result_df.to_csv("./exp-fl-3-1_time.csv", index=False)
    