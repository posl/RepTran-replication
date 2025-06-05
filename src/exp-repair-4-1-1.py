import os, sys, time, pickle, json, math
from tqdm import tqdm
from collections import defaultdict
from itertools import product
import numpy as np
import pandas as pd
import torch
from utils.helper import get_device, json2dict
from utils.vit_util import identfy_tgt_misclf, localize_neurons_with_mean_activation, get_batched_hs, get_batched_labels, ViTFromLastLayer, get_ori_model_predictions, maybe_initialize_repair_weights_
from utils.constant import ViTExperiment, ExperimentRepair1, Experiment3, ExperimentRepair2, Experiment1, Experiment4
from utils.log import set_exp_logging
from utils.arachne import calculate_top_n_flattened, calculate_bi_fi, calculate_pareto_front_flattened, approximate_pareto_front
from logging import getLogger
from datasets import load_from_disk
from transformers import ViTForImageClassification
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

logger = getLogger("base_logger")
device = get_device()

def main(ds_name, k, tgt_rank, misclf_type, fpfn, w_num, beta=None):
    print(f"ds_name: {ds_name}, fold_id: {k}, tgt_rank: {tgt_rank}, misclf_type: {misclf_type}, fpfn: {fpfn}")
    misclf_ptn = misclf_type if fpfn is None else f"{misclf_type}_{fpfn}"
    
    ts = time.perf_counter()
    
    # datasetをロード (true_labelsが欲しいので)
    ds_dirname = f"{ds_name}_fold{k}"
    ds = load_from_disk(os.path.join(ViTExperiment.DATASET_DIR, ds_dirname))
    if ds_name == "c10" or ds_name == "tiny-imagenet":
        label_col = "label"
    elif ds_name == "c100":
        label_col = "fine_label"
    else:
        raise NotImplementedError(ds_name)
    # ラベルの取得 (shuffleされない)
    labels = {
        "train": np.array(ds["train"][label_col]),
        "repair": np.array(ds["repair"][label_col]),
        "test": np.array(ds["test"][label_col])
    }
    tgt_pos = ViTExperiment.CLS_IDX
    
    # 結果とかログの保存先を先に作っておく
    # pretrained modelのディレクトリ
    pretrained_dir = getattr(ViTExperiment, ds_name.replace("-", "_")).OUTPUT_DIR.format(k=k)
    save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_weights_location")
    if misclf_type == "all":
        save_dir = os.path.join(pretrained_dir, f"all_weights_location")
    if fpfn is not None and misclf_type == "tgt":
        save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_{fpfn}_weights_location")
    os.makedirs(save_dir, exist_ok=True)

    # tgt_rankの誤分類情報を取り出す
    tgt_split = "repair" # NOTE: we only use repair split for repairing
    tgt_layer = 11 # NOTE: we only use the last layer for repairing
    misclf_info_dir = os.path.join(pretrained_dir, "misclf_info")
    misclf_pair, tgt_label, tgt_mis_indices = identfy_tgt_misclf(misclf_info_dir, tgt_split=tgt_split, tgt_rank=tgt_rank, misclf_type=misclf_type, fpfn=fpfn)
    
    # original model の repair setの各サンプルに対する正解/不正解のインデックスを取得
    pred_res_dir = os.path.join(pretrained_dir, "pred_results", "PredictionOutput")
    if misclf_type == "tgt":
        ori_pred_labels, _, indices_to_correct_tgt, _, indices_to_correct_others = get_ori_model_predictions(pred_res_dir, labels, tgt_split=tgt_split, misclf_type=misclf_type, tgt_label=tgt_label)
        # mis_clf=tgtでも全ての正解サンプルを選ぶ
        indices_to_correct = np.sort(np.concatenate([indices_to_correct_tgt, indices_to_correct_others]))
    else:
        ori_pred_labels, _, indices_to_correct = get_ori_model_predictions(pred_res_dir, labels, tgt_split=tgt_split, misclf_type=misclf_type, tgt_label=tgt_label)
    print(f"len(indices_to_correct): {len(indices_to_correct)}, len(tgt_mis_indices): {len(tgt_mis_indices)}")
    
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

    vscore_cor_dir = os.path.join(pretrained_dir, "vscores")
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
    places_to_neuron, tgt_neuron_score, neuron_scores = localize_neurons_with_mean_activation(vscore_before_dir, vscore_dir, vscore_after_dir, tgt_layer, n=None, intermediate_states=cached_mid_states, tgt_mis_indices=tgt_mis_indices, misclf_pair=misclf_pair, tgt_label=tgt_label, fpfn=fpfn, return_all_neuron_score=True, vscore_abs=True, covavg=False, vscore_cor_dir=vscore_cor_dir)
    # log表示
    # logger.info(f"places_to_neuron={places_to_neuron}")
    # logger.info(f"num(pos_to_fix)={len(places_to_neuron)}")
    # 位置情報を保存
    # print(f"len(places_to_neuron): {len(places_to_neuron)}")
    # print(f"tgt_neuron_score.shape: {tgt_neuron_score.shape}")
    # print(f"tgt_neuron_score: {tgt_neuron_score}")
    # print(f"neuron_scores.shape: {neuron_scores.shape}")
    # print(f"neuron_scores: {neuron_scores}")
    # print(np.min(neuron_scores), np.max(neuron_scores), np.mean(neuron_scores), np.std(neuron_scores))
    
    # ============================================================
    # ここまでで Vdiff x Use_i によるニューロンごとのスコア計算ができたので，次は勾配も使った重み特定をする．
    # ============================================================
    
    # キャッシュの保存用のディレクトリ
    cache_dir = os.path.join(pretrained_dir, f"cache_hidden_states_before_layernorm_{tgt_split}")
    cache_path = os.path.join(cache_dir, f"hidden_states_before_layernorm_{tgt_layer}.npy")
    # cache_pathに存在することを確認
    assert os.path.exists(cache_path), f"cache_path: {cache_path} does not exist."
    # vit_utilsの関数を使ってバッチを取得
    batch_size = ViTExperiment.BATCH_SIZE
    
    # 正解サンプル (I_pos) と誤りサンプル (I_neg) を分割
    correct_batched_hidden_states = get_batched_hs(cache_path, batch_size, indices_to_correct)
    correct_batched_labels = get_batched_labels(labels[tgt_split], batch_size, indices_to_correct)
    incorrect_batched_hidden_states = get_batched_hs(cache_path, batch_size, tgt_mis_indices)
    incorrect_batched_labels = get_batched_labels(labels[tgt_split], batch_size, tgt_mis_indices)
    
    # hidden_states_before_layernormのshapeを確認
    assert len(correct_batched_hidden_states) == len(correct_batched_labels), f"len(correct_batched_hidden_states): {len(correct_batched_hidden_states)}, len(correct_batched_labels): {len(correct_batched_labels)}"
    assert len(incorrect_batched_hidden_states) == len(incorrect_batched_labels), f"len(incorrect_batched_hidden_states): {len(incorrect_batched_hidden_states)}, len(incorrect_batched_labels): {len(incorrect_batched_labels)}"
    
    # ロスの勾配の取得に必要なモデルをロード
    model, loading_info = ViTForImageClassification.from_pretrained(pretrained_dir, output_loading_info=True)
    model.to(device).eval()
    model = maybe_initialize_repair_weights_(model, loading_info["missing_keys"])
    vit_from_last_layer = ViTFromLastLayer(model)
    vit_from_last_layer.eval()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
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
        tgt_mis_indices,
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
    # ここから
    # neuron_scoresを使って勾配とフォワードインパクトを調整
    print(f"Adjusting grad_loss and fwd_imp with neuron_scores...")
    for i in range(2):
        broadcasted_neuron_scores = neuron_scores[:, np.newaxis, ] if i == 0 else neuron_scores[np.newaxis, :]
        if beta is None:
            grad_loss_list[i] *= broadcasted_neuron_scores
            fwd_imp_list[i] *= broadcasted_neuron_scores
        else: # こっちはあんまりよくないかも
            grad_loss_list[i] *= (1.0 + beta * broadcasted_neuron_scores)
            fwd_imp_list[i] *= (1.0 + beta * broadcasted_neuron_scores)
    # 重みつきしてtopnの計算
    print(f"Calculating TopN for target weights...")
    identified_indices = calculate_top_n_flattened(grad_loss_list, fwd_imp_list, n=w_num)
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
    
    # 最終的に，location_save_pathに各中間ニューロンの重みの位置情報を保存する
    if fpfn is not None and misclf_type == "tgt":
        location_save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_{fpfn}_weights_location")
    elif misclf_type == "all":
        location_save_dir = os.path.join(pretrained_dir, f"all_weights_location")
    else:
        location_save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_weights_location")
    if beta is None:
        location_save_path = os.path.join(location_save_dir, f"exp-repair-4-1_location_n{w_num}_weight_ours.npy")
    else:
        location_save_path = os.path.join(location_save_dir, f"exp-repair-4-1_location_n{w_num}_beta{beta}_weight_ours.npy")
    np.save(location_save_path, (pos_before, pos_after))
    print(f"saved location information to {location_save_path}")
    
    # 終了時刻
    te = time.perf_counter()
    elapsed_time = te - ts
    return elapsed_time

if __name__ == "__main__":
    ds = "tiny-imagenet"
    # k_list = range(5)
    k_list = [0]
    tgt_rank_list = range(1, 4)
    misclf_type_list = ["src_tgt", "tgt"]
    fpfn_list = [None, "fp", "fn"]
    w_num_list = [236, 472, 944] # exp-repair-4.md 参照
    
    results = []
    for k, tgt_rank, misclf_type, fpfn, w_num in product(k_list, tgt_rank_list, misclf_type_list, fpfn_list, w_num_list):
        print(f"\nStart: ds={ds}, k={k}, tgt_rank={tgt_rank}, misclf_type={misclf_type}, fpfn={fpfn}, w_num={w_num} \n====================================================================")
        if (misclf_type == "src_tgt" or misclf_type == "all") and fpfn is not None: # misclf_type == "src_tgt" or "all"の時はfpfnはNoneだけでいい
            # print(f"Skipping: misclf_type={misclf_type} with fpfn={fpfn} is not valid.")
            continue
        if misclf_type == "tgt" and fpfn is None:
            # print(f"Skipping: misclf_type={misclf_type} with fpfn={fpfn} is not valid.")
            continue
        elapsed_time = main(ds, k, tgt_rank, misclf_type, fpfn, w_num=w_num)
        results.append({"ds": ds, "k": k, "tgt_rank": tgt_rank, "misclf_type": misclf_type, "fpfn": fpfn, "w_num": w_num, "elapsed_time": elapsed_time})
    # results を csv にして保存
    result_df = pd.DataFrame(results)
    result_df.to_csv(f"./exp-repair-4-1-1_time_{ds}.csv", index=False)