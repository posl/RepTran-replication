import os, sys, time, pickle, json
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import product
import torch
import torch.optim as optim
from datasets import load_from_disk
from transformers import ViTForImageClassification
from utils.helper import get_device, get_bottom3_keys_from_json
from utils.vit_util import transforms_c100, localize_neurons_with_mean_activation, get_batched_hs, get_batched_labels, ViTFromLastLayer, get_ori_model_predictions
from utils.arachne import calculate_bi_fi, calculate_top_n_flattened
from utils.constant import ViTExperiment, Experiment, ExperimentRepair1, Experiment1, ExperimentRepair2
from utils.log import set_exp_logging
from logging import getLogger

logger = getLogger("base_logger")

def main(n, beta):
    pretrained_dir = ViTExperiment.c100.OUTPUT_DIR.format(k=0)
    # Get device (cuda or cpu)
    device = get_device()
    # Get this Python file name
    this_file_name = os.path.basename(__file__).split(".")[0]
    logger = set_exp_logging(exp_dir=pretrained_dir, exp_name=this_file_name)
    label_col = "fine_label"
    tgt_pos = ViTExperiment.CLS_IDX
    batch_size = ViTExperiment.BATCH_SIZE

    # Load CIFAR-100-C dataset
    dataset_dir = Experiment.DATASET_DIR
    ds = load_from_disk(os.path.join(dataset_dir, "c100c"))
    labels = {
        key: np.array(ds[key][label_col]) for key in ds.keys()
    }
    ds_preprocessed = ds.with_transform(transforms_c100)
    # Load clean data (C100)
    ori_ds = load_from_disk(os.path.join(ViTExperiment.DATASET_DIR, "c100_fold0"))
    ori_labels = {
        "train": np.array(ori_ds["train"][label_col]),
        "repair": np.array(ori_ds["repair"][label_col]),
        "test": np.array(ori_ds["test"][label_col])
    }
    
    # Load pretrained model
    model = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
    model.eval()
    vit_from_last_layer = ViTFromLastLayer(model)
    vit_from_last_layer.eval()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # configuration
    tgt_layer = 11 # NOTE: we only use the last layer for repairing
    
    # モデルがクリーンデータで正解したサンプルのインデックスを取得
    pred_res_dir = os.path.join(pretrained_dir, "pred_results", "PredictionOutput")
    ori_pred_labels, is_correct, indices_to_correct = get_ori_model_predictions(pred_res_dir, ori_labels, tgt_split="repair", misclf_type=None, tgt_label=None)
    
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
        places_to_neuron, tgt_neuron_score, neuron_scores = localize_neurons_with_mean_activation(vscore_before_dir, vscore_dir, vscore_after_dir, tgt_layer, n=None, intermediate_states=cached_mid_states, tgt_mis_indices=None, return_all_neuron_score=True)
        print(f"neuron_scores.shape: {neuron_scores.shape}")
        print(f"neuron_scores: {neuron_scores}")

        # ============================================================
        # ここまでで Vdiff x Use_i によるニューロン特定ができたので，次は勾配も使った重み特定をする．
        # ============================================================
        
        # ノイズタイプkeyに対して間違ったサンプルを取得
        indices_to_incorrect = mis_indices_dict[key]
        
        # クリーンデータにおける正解時のキャッシュを取得
        cache_dir = os.path.join(pretrained_dir, f"cache_hidden_states_before_layernorm_repair")
        cache_path = os.path.join(cache_dir, f"hidden_states_before_layernorm_{tgt_layer}.npy")
        correct_batched_hidden_states = get_batched_hs(cache_path, batch_size, indices_to_correct)
        correct_batched_labels = get_batched_labels(ori_labels["repair"], batch_size, indices_to_correct)
        assert len(correct_batched_hidden_states) == len(correct_batched_labels), f"len(correct_batched_hidden_states): {len(correct_batched_hidden_states)}, len(correct_batched_labels): {len(correct_batched_labels)}"
        # ノイズデータにおける不正解時のキャッシュを取得
        cache_dir = os.path.join(pretrained_dir, f"cache_hidden_states_before_layernorm_{key}")
        cache_path = os.path.join(cache_dir, f"hidden_states_before_layernorm_{tgt_layer}.npy")
        incorrect_batched_hidden_states = get_batched_hs(cache_path, batch_size, indices_to_incorrect)
        incorrect_batched_labels = get_batched_labels(labels[key], batch_size, indices_to_incorrect)
        assert len(incorrect_batched_hidden_states) == len(incorrect_batched_labels), f"len(incorrect_batched_hidden_states): {len(incorrect_batched_hidden_states)}, len(incorrect_batched_labels): {len(incorrect_batched_labels)}"
        
        # =========================================
        # BIとFIの計算
        # =========================================
        
        # 全体の grad_loss と fwd_imp を統合
        grad_loss_list = [] # [Wbefに対するgrad_loss, Waftに対するgrad_loss]
        fwd_imp_list = []  # [Wbefに対するfwd_imp, Waftに対するfwd_imp]
        # 正解サンプルに対するBI, FI
        print(f"Calculating BI and FI... (correct samples)")
        pos_results = calculate_bi_fi(
            indices_to_correct,
            correct_batched_hidden_states,
            correct_batched_labels,
            vit_from_last_layer,
            optimizer,
            tgt_pos,
        )
        # 誤りサンプルに対するBI, FI
        print(f"Calculating BI and FI... (incorrect samples)")
        neg_results = calculate_bi_fi(
            indices_to_incorrect,
            incorrect_batched_hidden_states,
            incorrect_batched_labels,
            vit_from_last_layer,
            optimizer,
            tgt_pos,
        )
        
        # "before" と "after" のそれぞれで計算
        for ba in ["before", "after"]:
            # Gradient Loss (Arachne Algorithm1 L6)
            grad_loss = neg_results[ba]["bw"] / (1 + pos_results[ba]["bw"])
            print(f"{ba} - grad_loss.shape: {grad_loss.shape}")  # shape: (out_dim, in_dim)
            grad_loss_list.append(grad_loss)

            # Forward Impact (Arachne Algorithm1 L9)
            fwd_imp = neg_results[ba]["fw"] / (1 + pos_results[ba]["fw"])
            print(f"{ba} - fwd_imp.shape: {fwd_imp.shape}")  # shape: (out_dim, in_dim)
            fwd_imp_list.append(fwd_imp)

            # "before" の out_dim を取得
            if ba == "before":
                out_dim_before = grad_loss.shape[0]  # out_dim_before = out_dim

        # パレートフロントの計算
        print("Calculating top n for target weights...")
        identified_indices = calculate_top_n_flattened(grad_loss_list, fwd_imp_list, n=None)
        print(f"len(identified_indices['bef']): {len(identified_indices['bef'])}, len(identified_indices['aft']): {len(identified_indices['aft'])}")
        
        # "before" と "after" に分けて格納
        pos_before = identified_indices["bef"]
        pos_after = identified_indices["aft"]
        # 重みごとのスコア
        weighted_scores = identified_indices["scores"]
        assert len(weighted_scores) == len(pos_before) + len(pos_after), f"len(weighted_scores): {len(weighted_scores)}, len(pos_before): {len(pos_before)}, len(pos_after): {len(pos_after)}"

        # 結果の出力
        print(f"pos_before.shape: {pos_before.shape}, pos_after.shape: {pos_after.shape}")
        print(f"len(weighted scores): {len(weighted_scores)}")
        
        # こっからVdiffとBLの融合=====================
        # BLで計算した重みごとのスコアにVdiffの係数をかける
        wscore_gated = weighted_scores.copy()  # コピーして上書き（in-placeでゲーティング）
        
        num_wbef = len(pos_before)
        # -------------------------------
        # 1) Wbef 部分 (行=3072, 列=768)
        # 行が中間ニューロン -> neuron_scoresの軸
        # -------------------------------
        wbef_2d = wscore_gated[:num_wbef].reshape(3072, 768)
        # neuron_scores: shape=(3072,)
        # => neuron_scores[:, np.newaxis]: shape=(3072,1)
        # => ブロードキャストで (3072,768) に対応
        wbef_2d *= (1.0 + beta * neuron_scores[:, np.newaxis])

        # -------------------------------
        # 2) Waft 部分 (行=768, 列=3072)
        # 列が中間ニューロン -> neuron_scoresの軸
        # -------------------------------
        waft_2d = wscore_gated[num_wbef:].reshape(768, 3072)
        # neuron_scores[np.newaxis,:]: shape=(1,3072)
        # => ブロードキャストで (768,3072) に対応
        waft_2d *= (1.0 + beta * neuron_scores[np.newaxis, :])
        
        # print(len(wscore_gated)) # shape: (num_wbef + num_waft,)
        
        # スコアが高い順にソートして上位n件のインデックスを取得
        w_num = 8 * n * n
        top_n_indices = np.argsort(wscore_gated)[-w_num:][::-1]  # 降順で取得
        # befとaftに分類し、元の形状に戻す（ここが重要な修正ポイント）
        shape_bef = (3072, 768)
        shape_aft = (768, 3072)
        # befとaftに分類し、元の形状に戻す
        top_n_bef = np.array([
            np.unravel_index(idx, shape_bef) for idx in top_n_indices if idx < num_wbef
        ])
        top_n_aft = np.array([
            np.unravel_index(idx - num_wbef, shape_aft) for idx in top_n_indices if idx >= num_wbef
        ])
        identified_indices = {"bef": top_n_bef, "aft": top_n_aft, "scores": wscore_gated[top_n_indices]}
        # "before" と "after" に分けて格納
        pos_before = identified_indices["bef"]
        pos_after = identified_indices["aft"]
        print(f"pos_before.shape: {pos_before.shape}, pos_after.shape: {pos_after.shape}")
        print(f"len(weighted scores): {len(wscore_gated[top_n_indices])}")
        
        location_save_dir = os.path.join(pretrained_dir, f"corruptions_top{rank+1}", f"weights_location")
        os.makedirs(location_save_dir, exist_ok=True)
        location_save_path = os.path.join(location_save_dir, f"exp-c100c-fl-6_location_n{n}_beta{beta}_weight_ours.npy")
        np.save(location_save_path, (pos_before, pos_after))
        print(f"saved location information to {location_save_path}")
        # 終了時刻
        te = time.perf_counter()
        elapsed_time = te - ts
        ret_list.append({"n": n, "beta": beta, "tgt_rank": rank+1, "corruption": key, "elapsed_time": elapsed_time})
    return ret_list

        
if __name__ == "__main__":
    results = []
    n_list = [Experiment1.NUM_IDENTIFIED_WEIGHTS, ExperimentRepair1.NUM_IDENTIFIED_WEIGHTS, ExperimentRepair2.NUM_IDENTIFIED_WEIGHTS]
    beta_list = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    for n, beta in product(n_list, beta_list):
        print(f"\nStart experiment with n: {n}, beta: {beta}\n================================")
        ret_list = main(n, beta)
        results.extend(ret_list)
        print(results)
    # results を csv にしてSave
    result_df = pd.DataFrame(results)
    result_df.to_csv("./exp-c100c-fl-6_time.csv", index=False)