import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm
from utils.helper import get_device
device = get_device()

def calculate_top_n_flattened(grad_loss_list, fwd_imp_list, n=None, weight_grad_loss=0.5, weight_fwd_imp=0.5):
    """
    BI, FIを平滑化し、重み付き平均でスコアを計算して上位n件を取得。
    
    Args:
        grad_loss_list (list): BI のリスト [W_bef の BI, W_aft の BI]
        fwd_imp_list (list): FI のリスト [W_bef の FI, W_aft の FI]
        n (int): 上位n件を取得する数 (Noneの場合はソートせずに全件)
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
    
    # インデックスの決定
    if n is None:
        top_n_indices = np.arange(len(weighted_scores))  # ソートせず全件
    else:
        top_n_indices = np.argsort(weighted_scores)[-n:][::-1]  # 降順でn件

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
