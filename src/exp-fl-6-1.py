import os, sys, time, pickle, json, math
from tqdm import tqdm
from collections import defaultdict
from itertools import product
import numpy as np
import pandas as pd
import torch
from utils.helper import get_device, json2dict
from utils.vit_util import identfy_tgt_misclf, localize_neurons_with_mean_activation, get_batched_hs, get_batched_labels, ViTFromLastLayer, get_ori_model_predictions
from utils.constant import ViTExperiment, ExperimentRepair1, Experiment3, ExperimentRepair2, Experiment1, Experiment4
from utils.log import set_exp_logging
from utils.arachne import calculate_top_n_flattened, calculate_bi_fi
from logging import getLogger
from datasets import load_from_disk
from transformers import ViTForImageClassification
import torch
import torch.optim as optim

logger = getLogger("base_logger")
device = get_device()

def main(ds_name, k, tgt_rank, misclf_type, fpfn, n, beta=0.5):
    print(f"ds_name: {ds_name}, fold_id: {k}, tgt_rank: {tgt_rank}, misclf_type: {misclf_type}, fpfn: {fpfn}, n: {n}")
    
    ts = time.perf_counter()
    
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

    # tgt_rankの誤分類情報を取り出す
    tgt_split = "repair" # NOTE: we only use repair split for repairing
    tgt_layer = 11 # NOTE: we only use the last layer for repairing
    misclf_info_dir = os.path.join(pretrained_dir, "misclf_info")
    misclf_pair, tgt_label, tgt_mis_indices = identfy_tgt_misclf(misclf_info_dir, tgt_split=tgt_split, tgt_rank=tgt_rank, misclf_type=misclf_type, fpfn=fpfn)
    
    # original model の repair setの各サンプルに対する正解/不正解のインデックスを取得
    pred_res_dir = os.path.join(pretrained_dir, "pred_results", "PredictionOutput")
    if misclf_type == "tgt":
        ori_pred_labels, is_correct, indices_to_correct, is_correct_others, indices_to_correct_others = get_ori_model_predictions(pred_res_dir, labels, tgt_split=tgt_split, misclf_type=misclf_type, tgt_label=tgt_label)
    else:
        ori_pred_labels, is_correct, indices_to_correct = get_ori_model_predictions(pred_res_dir, labels, tgt_split=tgt_split, misclf_type=misclf_type, tgt_label=tgt_label)
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
    places_to_neuron, tgt_neuron_score, neuron_scores = localize_neurons_with_mean_activation(vscore_before_dir, vscore_dir, vscore_after_dir, tgt_layer, n=None, intermediate_states=cached_mid_states, tgt_mis_indices=tgt_mis_indices, misclf_pair=misclf_pair, tgt_label=tgt_label, fpfn=fpfn, return_all_neuron_score=True)
    # log表示
    # logger.info(f"places_to_neuron={places_to_neuron}")
    # logger.info(f"num(pos_to_fix)={len(places_to_neuron)}")
    # 位置情報を保存
    # print(f"len(places_to_neuron): {len(places_to_neuron)}")
    # print(f"tgt_neuron_score.shape: {tgt_neuron_score.shape}")
    # print(f"tgt_neuron_score: {tgt_neuron_score}")
    print(f"neuron_scores.shape: {neuron_scores.shape}")
    print(f"neuron_scores: {neuron_scores}")
    
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
    model = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
    model.eval()
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

    # forward/backward impacts の weighted sum
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
    
    # 最終的に，location_save_pathに各中間ニューロンの重みの位置情報を保存する
    if fpfn is not None and misclf_type == "tgt":
        location_save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_{fpfn}_weights_location")
    elif misclf_type == "all":
        location_save_dir = os.path.join(pretrained_dir, f"all_weights_location")
    else:
        location_save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_weights_location")
    location_save_path = os.path.join(location_save_dir, f"exp-fl-6_location_n{n}_beta{beta}_weight_ours.npy")
    np.save(location_save_path, (pos_before, pos_after))
    print(f"saved location information to {location_save_path}")
    # 終了時刻
    te = time.perf_counter()
    elapsed_time = te - ts
    return elapsed_time

if __name__ == "__main__":
    ds = "c100"
    # k_list = range(5)
    k_list = [0]
    tgt_rank_list = range(1, 6)
    # misclf_type_list = ["all", "src_tgt", "tgt"]
    misclf_type_list = ["src_tgt", "tgt"]
    fpfn_list = [None, "fp", "fn"]
    n_list = [Experiment4.NUM_IDENTIFIED_WEIGHTS]
    # n_list = [Experiment1.NUM_IDENTIFIED_WEIGHTS, ExperimentRepair1.NUM_IDENTIFIED_WEIGHTS, ExperimentRepair2.NUM_IDENTIFIED_WEIGHTS]
    n_str = "_".join([str(n) for n in n_list])
    beta_list = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    results = []
    for k, tgt_rank, misclf_type, fpfn, n, beta in product(k_list, tgt_rank_list, misclf_type_list, fpfn_list, n_list, beta_list):
        print(f"\nStart: ds={ds}, k={k}, n={n}, beta={beta}, tgt_rank={tgt_rank}, misclf_type={misclf_type}, fpfn={fpfn}\n====================================================================")
        if (misclf_type == "src_tgt" or misclf_type == "all") and fpfn is not None: # misclf_type == "src_tgt" or "all"の時はfpfnはNoneだけでいい
            continue
        if misclf_type == "all" and tgt_rank != 1: # misclf_type == "all"の時にtgt_rankは関係ないのでこのループもスキップすべき
            continue
        elapsed_time = main(ds, k, tgt_rank, misclf_type, fpfn, n=n, beta=beta)
        results.append({"ds": ds, "k": k, "n": n, "beta": beta, "tgt_rank": tgt_rank, "misclf_type": misclf_type, "fpfn": fpfn, "elapsed_time": elapsed_time})
    # results を csv にして保存
    result_df = pd.DataFrame(results)
    result_df.to_csv(f"./exp-fl-6-1_time_n{n_str}.csv", index=False)