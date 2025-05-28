import os, sys, time, pickle, json, math
from tqdm import tqdm
from collections import defaultdict
from itertools import product
import numpy as np
import pandas as pd
from utils.helper import get_device, json2dict
from utils.vit_util import transforms, transforms_c100, ViTFromLastLayer, identfy_tgt_misclf, get_ori_model_predictions, get_new_model_predictions, get_batched_hs, get_batched_labels, sample_from_correct_samples, sample_true_positive_indices_per_class, maybe_initialize_repair_weights_

from utils.constant import ViTExperiment, Experiment1, ExperimentRepair1, ExperimentRepair2, Experiment4
from utils.log import set_exp_logging
from logging import getLogger
from datasets import load_from_disk
from transformers import ViTForImageClassification
import torch
import torch.optim as optim

# デバイス (cuda, or cpu) の取得
device = get_device()

def calculate_top_n_flattened(grad_loss_list, fwd_imp_list, n, weight_grad_loss=0.5, weight_fwd_imp=0.5):
    """
    BI, FIを平滑化し、重み付き平均でスコアを計算して上位n件を取得。
    
    Args:
        grad_loss_list (list): BI のリスト [W_bef の BI, W_aft の BI]
        fwd_imp_list (list): FI のリスト [W_bef の FI, W_aft の FI]
        n (int): 上位n件を取得する数
        weight_grad_loss (float): grad_loss の重み (0~1の範囲)
        weight_fwd_imp (float): fwd_imp の重み (0~1の範囲)
    
    Returns:
        dict: 上位n件のインデックス {"bef": [...], "aft": [...]} とそのスコア
    """
    # BI, FIを一列に変換
    flattened_grad_loss = torch.cat([x.flatten() for x in grad_loss_list])
    flattened_fwd_imp = torch.cat([x.flatten() for x in fwd_imp_list])

    # befとaftの形状を取得
    shape_bef = grad_loss_list[0].shape
    shape_aft = grad_loss_list[1].shape

    # befとaftの境目インデックスを記録
    split_idx = grad_loss_list[0].numel()
    
    # 正規化
    grad_loss_min, grad_loss_max = flattened_grad_loss.min(), flattened_grad_loss.max()
    fwd_imp_min, fwd_imp_max = flattened_fwd_imp.min(), flattened_fwd_imp.max()

    normalized_grad_loss = (flattened_grad_loss - grad_loss_min) / (grad_loss_max - grad_loss_min + 1e-8)
    normalized_fwd_imp = (flattened_fwd_imp - fwd_imp_min) / (fwd_imp_max - fwd_imp_min + 1e-8)

    # 重み付きスコアを計算
    weighted_scores = (
        weight_grad_loss * normalized_grad_loss +
        weight_fwd_imp * normalized_fwd_imp
    ).detach().cpu().numpy()

    # スコアが高い順にソートして上位n件のインデックスを取得
    top_n_indices = np.argsort(weighted_scores)[-n:][::-1]  # 降順で取得

    # befとaftに分類し、元の形状に戻す
    top_n_bef = np.array([
        np.unravel_index(idx, shape_bef) for idx in top_n_indices if idx < split_idx
    ])
    top_n_aft = np.array([
        np.unravel_index(idx - split_idx, shape_aft) for idx in top_n_indices if idx >= split_idx
    ])

    return {"bef": top_n_bef, "aft": top_n_aft, "scores": weighted_scores[top_n_indices]}


def calculate_pareto_front_flattened(grad_loss_list, fwd_imp_list):
    """
    BI, FIを平滑化してパレートフロントを計算
    Args:
        grad_loss_list (list): BI のリスト [W_bef の BI, W_aft の BI]
        fwd_imp_list (list): FI のリスト [W_bef の FI, W_aft の FI]
    Returns:
        dict: パレートフロントのインデックス {"bef": [...], "aft": [...]}
    """
    # BI, FIを一列に変換
    flattened_grad_loss = torch.cat([x.flatten() for x in grad_loss_list])
    flattened_fwd_imp = torch.cat([x.flatten() for x in fwd_imp_list])
    
    # befとaftの形状を取得
    shape_bef = grad_loss_list[0].shape
    shape_aft = grad_loss_list[1].shape

    # befとaftの境目インデックスを記録
    split_idx = grad_loss_list[0].numel()

    # パレートフロントを計算
    pareto_indices = approximate_pareto_front(flattened_grad_loss, flattened_fwd_imp)

    # befとaftに分類し、元の形状に戻す
    pareto_bef = np.array([
        np.unravel_index(idx, shape_bef) for idx in pareto_indices if idx < split_idx
    ])
    pareto_aft = np.array([
        np.unravel_index(idx - split_idx, shape_aft) for idx in pareto_indices if idx >= split_idx
    ])

    return {"bef": pareto_bef, "aft": pareto_aft}


def approximate_pareto_front(flattened_bi_values, flattened_fi_values):
    """
    平滑化されたデータからパレートフロントを計算
    Args:
        flattened_bi_values (torch.Tensor): フラット化された BI
        flattened_fi_values (torch.Tensor): フラット化された FI
    Returns:
        list: パレートフロントに含まれるインデックスリスト
    """
    # BI, FIをnumpyに変換
    bi_values = flattened_bi_values.detach().cpu().numpy()
    fi_values = flattened_fi_values.detach().cpu().numpy()

    # BI, FIを2次元の点として結合
    points = np.stack([bi_values, fi_values], axis=1)

    # パレートフロントを計算
    pareto_mask = np.ones(points.shape[0], dtype=bool)
    for i, point in enumerate(points):
        if pareto_mask[i]:
            pareto_mask[pareto_mask] &= ~(
                np.all(points[pareto_mask] <= point, axis=1) &
                np.any(points[pareto_mask] < point, axis=1)
            )

    pareto_indices = np.where(pareto_mask)[0]
    return pareto_indices


def calculate_bi_fi(indices, batched_hidden_states, batched_labels, vit_from_last_layer, optimizer, tgt_pos):
    """
    指定されたサンプル集合（正解または誤り）に対して BI と FI を計算し、before/after に分ける。
    Args:
        indices (list): サンプルのインデックス集合
        batched_hidden_states (list): キャッシュされたバッチごとの hidden states
        batched_labels (list): バッチごとの正解ラベル
        vit_from_last_layer (ViTFromLastLayer): ViTモデルの最終層ラッパー
        optimizer (torch.optim.Optimizer): PyTorchのオプティマイザ
        tgt_pos (int): ターゲットポジション（通常CLSトークン）
    Returns:
        defaultdict: {"before": {"bw": grad_bw, "fw": grad_fw}, "after": {"bw": grad_bw, "fw": grad_fw}}
    """
    # results = defaultdict(lambda: {"bw": [], "fw": []})
    results = defaultdict(lambda: {"bw": None, "fw": None, "count": 0})  # 集計用


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

        # ForwardImpact計算用データ
        cached_state_aft_ln = vit_from_last_layer.base_model_last_layer.layernorm_after(cached_state)
        cached_state_aft_mid = vit_from_last_layer.base_model_last_layer.intermediate(cached_state_aft_ln)
        cached_state_aft_ln = cached_state_aft_ln[:, tgt_pos, :]
        cached_state_aft_mid = cached_state_aft_mid[:, tgt_pos, :]

        # BackwardImpact (BI) と ForwardImpact (FI) の計算
        for cs, tgt_component, ba_layer in zip(
            [cached_state_aft_ln, cached_state_aft_mid],
            [vit_from_last_layer.base_model_last_layer.intermediate.dense, vit_from_last_layer.base_model_last_layer.output.dense],
            ["before", "after"],
        ):
            # BI: ロスの勾配
            grad_bw = tgt_component.weight.grad.cpu()
            # print(f"{ba_layer} - grad_bw.shape: {grad_bw.shape}")  # shape: (out_dim, in_dim)
            if results[ba_layer]["bw"] is None:
                results[ba_layer]["bw"] = grad_bw.detach().clone().cpu()
            else:
                results[ba_layer]["bw"] += grad_bw.detach().cpu()

            # FI: logits の勾配 × 正規化されたニューロンの重み
            grad_out_weight = torch.autograd.grad(
                logits, tgt_component.weight, grad_outputs=torch.ones_like(logits), retain_graph=True
            )[0]
            tgt_weight_expanded = tgt_component.weight.unsqueeze(0)
            oi_expanded = cs.unsqueeze(1)
            
            # **までGPUで計算
            impact_out_weight = tgt_weight_expanded * oi_expanded
            normalization_terms = impact_out_weight.sum(dim=2)
            normalized_impact_out_weight = impact_out_weight / (normalization_terms[:, :, None] + 1e-8)
            mean_normalized_impact_out_weight = normalized_impact_out_weight.mean(dim=0)
            grad_fw = (mean_normalized_impact_out_weight * grad_out_weight).cpu() # ** ここでCPUに戻す
            # print(f"{ba_layer} - grad_fw.shape: {grad_fw.shape}")  # shape: (out_dim, in_dim)
            if results[ba_layer]["fw"] is None:
                results[ba_layer]["fw"] = grad_fw.detach().clone().cpu()
            else:
                results[ba_layer]["fw"] += grad_fw.detach().cpu()

            # カウントを更新
            results[ba_layer]["count"] += 1
    
    # バッチ全体の平均を計算
    for ba_layer in ["before", "after"]:
        if results[ba_layer]["count"] > 0:
            results[ba_layer]["bw"] = results[ba_layer]["bw"] / results[ba_layer]["count"]
            results[ba_layer]["fw"] = results[ba_layer]["fw"] / results[ba_layer]["count"]
    return results

def main(ds_name, k, tgt_rank, misclf_type, fpfn, sample_from_correct=False):
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
    
    # pretrained modelのロード
    pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR.format(k=k)
    # location informationの保存先
    # model = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
    # model.eval()
    model, loading_info = ViTForImageClassification.from_pretrained(pretrained_dir, output_loading_info=True)
    model.to(device).eval()
    model = maybe_initialize_repair_weights_(model, loading_info["missing_keys"])
    vit_from_last_layer = ViTFromLastLayer(model)
    vit_from_last_layer.eval()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # tgt_rankの誤分類情報を取り出す
    misclf_info_dir = os.path.join(pretrained_dir, "misclf_info")
    _, tgt_label, tgt_mis_indices = identfy_tgt_misclf(misclf_info_dir, tgt_split=tgt_split, tgt_rank=tgt_rank, misclf_type=misclf_type, fpfn=fpfn)
    indices_to_incorrect = tgt_mis_indices
    
    # original model の repair setの各サンプルに対する正解/不正解のインデックスを取得
    pred_res_dir = os.path.join(pretrained_dir, "pred_results", "PredictionOutput")
    if misclf_type == "tgt":
        ori_pred_labels, _, indices_to_correct_tgt, _, indices_to_correct_others = get_ori_model_predictions(pred_res_dir, labels, tgt_split=tgt_split, misclf_type=misclf_type, tgt_label=tgt_label)
        # mis_clf=tgtでも全ての正解サンプルを選ぶ
        indices_to_correct = np.sort(np.concatenate([indices_to_correct_tgt, indices_to_correct_others]))
    else:
        ori_pred_labels, _, indices_to_correct = get_ori_model_predictions(pred_res_dir, labels, tgt_split=tgt_split, misclf_type=misclf_type, tgt_label=tgt_label)
    print(f"len(indices_to_correct): {len(indices_to_correct)}, len(indices_to_incorrect): {len(indices_to_incorrect)}")
    
    # 正解データからrepairに使う一定数だけランダムに取り出す
    if sample_from_correct:
        # サンプルする場合
        sampled_indices_to_correct = sample_from_correct_samples(len(indices_to_incorrect), indices_to_correct)
    else:
        # サンプルしない場合
        sampled_indices_to_correct = indices_to_correct
    # 抽出した正解データと，全不正解データを結合して1つのデータセットにする
    tgt_indices = sampled_indices_to_correct.tolist() + indices_to_incorrect.tolist() # .tolist() は 非破壊的method
    # tgt_indicesは全てユニークな値であることを保証
    assert len(tgt_indices) == len(set(tgt_indices)), f"len(tgt_indices): {len(tgt_indices)}, len(set(tgt_indices)): {len(set(tgt_indices))}"
    print(f"len(tgt_indices): {len(tgt_indices)})")
    # tgt_indicesに対応するデータトラベルを取り出す
    tgt_labels = labels[tgt_split][tgt_indices]
    # FLに使う各サンプルの予測ラベルと正解ラベルを表示
    print(f"ori_pred_labels[tgt_indices]: {ori_pred_labels[tgt_indices]} (len: {len(ori_pred_labels[tgt_indices])})")
    print(f"ori_tgt_labels[tgt_indices]: {tgt_labels} (len: {len(tgt_labels)})")
    
    # キャッシュの保存用のディレクトリ
    cache_dir = os.path.join(pretrained_dir, f"cache_hidden_states_before_layernorm_{tgt_split}")
    cache_path = os.path.join(cache_dir, f"hidden_states_before_layernorm_{tgt_layer}.npy")
    # cache_pathに存在することを確認
    assert os.path.exists(cache_path), f"cache_path: {cache_path} does not exist."
    # vit_utilsの関数を使ってバッチを取得
    batch_size = ViTExperiment.BATCH_SIZE
    
    # 正解サンプル (I_pos) と誤りサンプル (I_neg) を分割
    correct_batched_hidden_states = get_batched_hs(cache_path, batch_size, sampled_indices_to_correct)
    correct_batched_labels = get_batched_labels(labels[tgt_split], batch_size, sampled_indices_to_correct)
    incorrect_batched_hidden_states = get_batched_hs(cache_path, batch_size, indices_to_incorrect)
    incorrect_batched_labels = get_batched_labels(labels[tgt_split], batch_size, indices_to_incorrect)
    
    assert len(correct_batched_hidden_states) == len(correct_batched_labels), f"len(correct_batched_hidden_states): {len(correct_batched_hidden_states)}, len(correct_batched_labels): {len(correct_batched_labels)}"
    
    # =========================================
    # BIとFIの計算
    # =========================================
    
    # 全体の grad_loss と fwd_imp を統合
    grad_loss_list = [] # [Wbefに対するgrad_loss, Waftに対するgrad_loss]
    fwd_imp_list = []  # [Wbefに対するfwd_imp, Waftに対するfwd_imp]
    # 正解サンプルに対するBI, FI
    print(f"Calculating BI and FI... (correct samples)")
    pos_results = calculate_bi_fi(
        sampled_indices_to_correct,
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
    print(f"Calculating Pareto front for target weights...")
    identified_indices = calculate_pareto_front_flattened(grad_loss_list, fwd_imp_list)
    print(f"len(identified_indices['bef']): {len(identified_indices['bef'])}, len(identified_indices['aft']): {len(identified_indices['aft'])}")
    
    # "before" と "after" に分けて格納
    pos_before = identified_indices["bef"]
    pos_after = identified_indices["aft"]
    
    # 結果の出力
    print(f"pos_before: {pos_before}")
    print(f"pos_after: {pos_after}")
    
    # 最終的に，location_save_pathに各中間ニューロンの重みの位置情報を保存する
    if fpfn is not None and misclf_type == "tgt":
        location_save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_{fpfn}_weights_location")
    elif misclf_type == "all":
        location_save_dir = os.path.join(pretrained_dir, f"all_weights_location")
    else:
        location_save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_weights_location")
    # old_location_save_path = os.path.join(location_save_dir, f"exp-fl-2_location_weight_bl.npy")
    location_save_path = os.path.join(location_save_dir, f"exp-fl-2_location_pareto_weight_bl.npy")
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
    tgt_rank_list = range(1, 4)
    # tgt_rank_list = range(1, 6)
    misclf_type_list = ["src_tgt", "tgt"]
    # misclf_type_list = ["all", "src_tgt", "tgt"]
    fpfn_list = [None, "fp", "fn"]
    results = []
    for k, tgt_rank, misclf_type, fpfn in product(k_list, tgt_rank_list, misclf_type_list, fpfn_list):
        if (misclf_type == "src_tgt" or misclf_type == "all") and fpfn is not None: # misclf_type == "src_tgt" or "all"の時はfpfnはNoneだけでいい
            # print(f"Skipping: misclf_type={misclf_type} with fpfn={fpfn} is not valid.")
            continue
        if misclf_type == "tgt" and fpfn is None:
            # print(f"Skipping: misclf_type={misclf_type} with fpfn={fpfn} is not valid.")
            continue
        print("\n")
        print("=" * 80)
        print(f"Start: ds={ds}, k={k}, tgt_rank={tgt_rank}, misclf_type={misclf_type}, fpfn={fpfn}")
        print("=" * 80)
        elapsed_time = main(ds, k, tgt_rank, misclf_type, fpfn)
        results.append({"ds": ds, "k": k, "tgt_rank": tgt_rank, "misclf_type": misclf_type, "fpfn": fpfn, "elapsed_time": elapsed_time})
    # results を csv にして保存
    result_df = pd.DataFrame(results)
    result_df.to_csv(f"./exp-repair-3-1-1_time_pareto.csv", index=False)