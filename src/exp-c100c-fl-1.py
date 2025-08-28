import os, sys, time, pickle, json
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict
import torch
import torch.optim as optim
from datasets import load_from_disk
from transformers import ViTForImageClassification
from utils.helper import get_device, get_bottom3_keys_from_json
from utils.vit_util import transforms_c100, localize_neurons_with_mean_activation, get_batched_hs, get_batched_labels, ViTFromLastLayer
from utils.constant import ViTExperiment, Experiment, ExperimentRepair1, Experiment3, ExperimentRepair2
from utils.log import set_exp_logging
from logging import getLogger

logger = getLogger("base_logger")

def main(n_ratio, w_num):
    pretrained_dir = ViTExperiment.c100.OUTPUT_DIR.format(k=0)
    # Get device (cuda or cpu)
    device = get_device()
    # Get this Python file name
    this_file_name = os.path.basename(__file__).split(".")[0]
    logger = set_exp_logging(exp_dir=pretrained_dir, exp_name=this_file_name)
    label_col = "fine_label"
    tgt_pos = ViTExperiment.CLS_IDX

    # Load CIFAR-100-C dataset
    dataset_dir = Experiment.DATASET_DIR
    ds = load_from_disk(os.path.join(dataset_dir, "c100c"))
    labels = {
        key: np.array(ds[key][label_col]) for key in ds.keys()
    }
    ds_preprocessed = ds.with_transform(transforms_c100)
    
    # Load pretrained model
    model = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
    model.eval()
    # configuration
    tgt_layer = 11 # NOTE: we only use the last layer for repairing
    
    # Only process bottom3 noise types by accuracy
    bottom3_keys = get_bottom3_keys_from_json(os.path.join(pretrained_dir, "corruption_accuracy.json"))
    
    # ノイズタイプごとの誤ったサンプルのインデックスを取得
    with open(os.path.join(pretrained_dir, "corruption_error_indices.json"), 'r') as f:
        mis_indices_dict = json.load(f)
        mis_indices_dict = {k: v for k, v in mis_indices_dict.items() if k in bottom3_keys}
    
    # vscoresのSaveディレクトリ
    vscore_before_dir = os.path.join(pretrained_dir, "vscores_before") # 使わないので実際はなくていい
    vscore_dir = os.path.join(pretrained_dir, "vscores")
    vscore_after_dir = os.path.join(pretrained_dir, "vscores_after") # 使わないので実際はなくていい
    
    
    # ノイズタイプごとの誤ったサンプルのインデックスを取得
    with open(os.path.join(pretrained_dir, "corruption_error_indices.json"), 'r') as f:
        mis_indices_dict = json.load(f)
        mis_indices_dict = {k: v for k, v in mis_indices_dict.items() if k in bottom3_keys}
    
    ret_list = []
    
    for rank, key in enumerate(bottom3_keys):
        ts = time.perf_counter()
        print(f"Processing {key}...")
        
        # 中間ニューロン値のキャッシュのロード
        mid_cache_dir = os.path.join(pretrained_dir, f"cache_states_{key}")
        mid_save_path = os.path.join(mid_cache_dir, f"intermediate_states_l{tgt_layer}_mis.pt")
        cached_mid_states = torch.load(mid_save_path, map_location="cpu")
        # cached_mid_statesをnumpy配列にする
        cached_mid_states = cached_mid_states.detach().numpy().copy() # (keyにおける誤ったサンプル数, 3072)
        print(f"cached_mid_states.shape: {cached_mid_states.shape}")
        
        # 結果Save用のディレクトリ
        save_dir = os.path.join(pretrained_dir, f"corruptions_top{rank+1}", f"weights_location")
        os.makedirs(save_dir, exist_ok=True)
        location_save_path = os.path.join(save_dir, f"exp-c100c-fl-1_location_n{n_ratio}_w{w_num}_neuron.npy")
        places_to_neuron, tgt_neuron_score = localize_neurons_with_mean_activation(vscore_before_dir, vscore_dir, vscore_after_dir, tgt_layer, n_ratio, intermediate_states=cached_mid_states, tgt_mis_indices=None)
        # Display logs
        logger.info(f"places_to_neuron={places_to_neuron}")
        logger.info(f"num(pos_to_fix)={len(places_to_neuron)}")
        # 位置情報をSave
        np.save(location_save_path, places_to_neuron)
        logger.info(f"saved location information to {location_save_path}")
        print(f"saved location information to {location_save_path}")
        print(f"len(places_to_neuron): {len(places_to_neuron)}")
        print(f"tgt_neuron_score.shape: {tgt_neuron_score.shape}")
        print(f"tgt_neuron_score: {tgt_neuron_score}")

        # ここまでで Vdiff x Use_i によるニューロン特定ができたので，次は勾配も使った重み特定をする．
    
        # キャッシュのSave用のディレクトリ
        cache_dir = os.path.join(pretrained_dir, f"cache_hidden_states_before_layernorm_{key}")
        cache_path = os.path.join(cache_dir, f"hidden_states_before_layernorm_{tgt_layer}.npy")
        # cache_pathに存在することを確認
        assert os.path.exists(cache_path), f"cache_path: {cache_path} does not exist."
        # vit_utilsの関数を使ってバッチを取得
        batch_size = ViTExperiment.BATCH_SIZE
        
        # 誤りサンプル (I_neg) を取得
        incorrect_batched_hidden_states = get_batched_hs(cache_path, batch_size, mis_indices_dict[key])
        incorrect_batched_labels = get_batched_labels(labels[key], batch_size, mis_indices_dict[key])
        assert len(incorrect_batched_hidden_states) == len(incorrect_batched_labels), "incorrect_batched_hidden_states and incorrect_batched_labels should have the same length."
        
        # ロスの勾配の取得に必要なモデルをロード
        model = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
        model.eval()
        vit_from_last_layer = ViTFromLastLayer(model)
        vit_from_last_layer.eval()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        
        # 結果のSave用
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

        # 位置情報をSave
        location_save_path = os.path.join(save_dir, f"exp-c100c-fl-1_location_n{n_ratio}_w{w_num}_weight.npy")
        np.save(location_save_path, (pos_before, pos_after))
        print(f"saved location information to {location_save_path}")
        # 終了時刻
        te = time.perf_counter()
        elapsed_time = te - ts
        ret_list.append({"tgt_rank": rank+1, "corruption": key, "elapsed_time": elapsed_time})
    return ret_list
        
        
if __name__ == "__main__":
    results = []
    exp_list = [Experiment3, ExperimentRepair1, ExperimentRepair2]
    
    for exp in exp_list:
        n_ratio, w_num = exp.NUM_IDENTIFIED_NEURONS_RATIO, exp.NUM_IDENTIFIED_WEIGHTS
        w_num = 8 * w_num * w_num
        print(f"Start experiment with n_ratio: {n_ratio}, w_num: {w_num}")
        ret_list = main(n_ratio, w_num)
        # ret_listの各dictにn_ratioとw_numを追加
        for ret in ret_list:
            ret["n_ratio"] = n_ratio
            ret["w_num"] = w_num
        results.extend(ret_list)
        print(results)
    # results を csv にしてSave
    result_df = pd.DataFrame(results)
    result_df.to_csv("./exp-c100c-fl-1_time.csv", index=False)