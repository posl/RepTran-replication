import os
import numpy as np
import pickle
import torch
import torch.nn as nn
from collections import defaultdict, Counter
from itertools import product
from transformers import ViTImageProcessor, Trainer
import sys
sys.path.append('../')
from utils.constant import ViTExperiment
import evaluate

processor = ViTImageProcessor.from_pretrained(ViTExperiment.ViT_PATH)
met_acc = evaluate.load("accuracy")
met_f1 = evaluate.load("f1")

def transforms(batch):
    """
    Preprocess image batch
    Applicable when the column name representing labels is label (c10)
    
    Parameters
    ------------------
    
    Returns
    ------------------
    
    """
    # Convert image batch to torch.tensor
    inputs = processor(images=batch["img"], return_tensors="pt")

    # Add label field during preprocessing
    inputs["labels"] = batch["label"]
    return inputs

def transforms_c100(batch):
    """
    Preprocess image batch
    Applicable when the column name representing labels is fine_label (c100)
    
    Parameters
    ------------------
    
    Returns
    ------------------
    
    """
    # Convert image batch to torch.tensor
    inputs = processor(images=batch["img"], return_tensors="pt")

    # Add label field during preprocessing
    inputs["labels"] = batch["fine_label"]
    
    if "ori_correct" in batch:
        inputs["ori_correct"] = batch["ori_correct"]
    return inputs

def pred_to_proba(pred):
    proba = torch.nn.functional.softmax(torch.tensor(pred.predictions), dim=-1)
    return proba.cpu().numpy()

def pred_to_labels(pred):
    # probaに変換されたnumpy配列を受け取る場合
    if isinstance(pred, np.ndarray):
        proba = pred
        labels = np.argmax(proba, axis=-1)
        return labels
    proba = torch.nn.functional.softmax(torch.tensor(pred.predictions), dim=-1)
    labels = torch.argmax(proba, dim=-1)
    return labels.cpu().numpy()

def compute_metrics(eval_pred):
    """
    予測結果から評価指標を計算する
    
    Parameters
    ------------------
    
    Returns
    ------------------
    
    """
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    predictions = np.argmax(logits, axis=-1)
    acc = met_acc.compute(predictions=predictions, references=labels)
    f1 = met_f1.compute(predictions=predictions, references=labels, average="macro")
    return {
        "accuracy": acc,
        "f1": f1,
    }

def count_pred_change(old_pred, new_pred):
    """
    修正前後のモデルの予測結果を比較して, 修正されたサンプル数, 壊れたサンプルを特定する
    
    Parameters
    ------------------
    old_pred: PredictionOutput
        修正前のモデルの予測結果
    new_pred: PredictionOutput
        修正後のモデルの予測結果

    Returns
    ------------------
    result: dict
        4種類の修正結果 (repaired, broken, non-repaired, non-broken) を示した辞書.
        キーが修正結果の名前，値がそのインデックスのリスト.
        NOTE: old_pred, new_predは同じ評価用データに対する予測結果で, shuffleもされていない前提.
    """
    old_labels = pred_to_labels(old_pred)
    new_labels = pred_to_labels(new_pred)
    true_labels = old_pred.label_ids
    # 4種類の判定を行う
    # 1. 修正されたサンプル
    repaired = np.where((old_labels != true_labels) & (new_labels == true_labels))[0]
    # 2. 壊れたサンプル
    broken = np.where((old_labels == true_labels) & (new_labels != true_labels))[0]
    # 3. 修正されていないサンプル
    non_repaired = np.where((old_labels != true_labels) & (new_labels != true_labels))[0]
    # 4. 壊れていないサンプル
    non_broken = np.where((old_labels == true_labels) & (new_labels == true_labels))[0]
    result = {
        "repaired": repaired,
        "broken": broken,
        "non_repaired": non_repaired,
        "non_broken": non_broken
    }
    return result

def get_vscore(batch_neuron_values, abs=True, covavg=True):
    """
    ニューロンに対するvscoreを返す
    
    Parameters
    ------------------
    batch_neuron_values: numpy.ndarray
        ニューロンの値を表す行列 (num_samples, num_neurons_of_tgt_layer)

    Returns
    ------------------
    vscore: numpy.ndarray
        ニューロンごとのvscore (num_neurons_of_tgt_layer, )
    """
    # num_samplesが1以下の場合は, (num_neurons_of_tgt_layer, ) の形状のnanを返す
    if batch_neuron_values.shape[0] <= 1:
        return np.full(batch_neuron_values.shape[1], np.nan)
    neuron_cov = np.cov(batch_neuron_values, rowvar=False) # (num_neurons_of_tgt_layer, num_neurons_of_tgt_layer)
    if abs:
        # adding covariances may cancel the effect, so take absolute value
        neuron_cov = np.abs(neuron_cov)
        num_neg = np.sum(neuron_cov < 0) # マイナス要素の数を取得
        assert num_neg == 0, f"Error: {num_neg} negative elements in neuron_cov"
    # ニューロン分散共分散行列の対角成分 = 各ニューロンの分散 を取得
    neuron_var = np.diag(neuron_cov)
    # neuron_covの各行の和
    neuron_cov_sum = np.nansum(neuron_cov, axis=0) # 自分の分散 + (他の共分散の総和)
    # 他ニューロンとの共分散の平均
    if covavg:
        cov_term = (neuron_cov_sum - neuron_var) / (neuron_cov_sum.shape[0] - 1)
    else:
        cov_term = neuron_cov_sum - neuron_var
    
    # vscoreを計算
    vscore = neuron_var + cov_term # (num_neurons_of_tgt_layer,)
    # vscoreの最小と最大を表示
    print(f"vscore min: {np.min(vscore)}, max: {np.max(vscore)}")
    return vscore

def localize_neurons(vscore_before_dir, vscore_dir, vscore_after_dir, tgt_layer, n, tgt_split="repair", misclf_pair=None, tgt_label=None, fpfn=None, rank_type="abs"):
    vmap_dic = defaultdict(np.array)
    for cor_mis in ["cor", "mis"]:
        ds_type = f"ori_{tgt_split}"
        # vscore_save_pathの設定
        vscore_save_path = os.path.join(vscore_dir, f"vscore_l1tol12_all_label_{ds_type}_{cor_mis}.npy")
        if misclf_pair is not None and cor_mis == "mis":
            # misclf_pairが指定されている場合は，その対象のデータのみを取得
            assert len(misclf_pair) == 2, f"Error: {misclf_pair}"
            slabel, tlabel = misclf_pair
            vscore_save_path = os.path.join(vscore_dir, f"vscore_l1tol12_{slabel}to{tlabel}_{ds_type}_{cor_mis}.npy")
        if tgt_label is not None and cor_mis == "mis":
            # tgt_labelが指定されている場合は，その対象のデータのみを取得
            vscore_save_path = os.path.join(vscore_dir, f"vscore_l1tol12_{tgt_label}_{ds_type}_{cor_mis}.npy")
            if fpfn is not None:
                vscore_save_path = os.path.join(vscore_dir, f"vscore_l1tol12_{tgt_label}_{ds_type}_{fpfn}_{cor_mis}.npy")
        # vscoreを読み込む
        vscores = np.load(vscore_save_path)
        vmap_dic[cor_mis] = vscores.T
    vmap_cor = vmap_dic["cor"]
    vmap_mis = vmap_dic["mis"]
    vmap_diff = vmap_cor - vmap_mis
    # vmap_diff[:, tgt_layer]の絶対値の上位n個を取得
    vmap_diff_abs = np.abs(vmap_diff[:, tgt_layer])
    if isinstance(n, int):
        top_idx = np.argsort(vmap_diff_abs)[::-1][:n] # top_idx[k] = vmap_diff_absの中でk番目に大きい値のインデックス
    elif isinstance(n, float):
        assert n <= 1, f"Error: {n}"
        num_neurons = vmap_diff_abs.shape[0]
        top_idx = np.argsort(vmap_diff_abs)[::-1][:int(num_neurons * n)] # top_idx[k] = vmap_diff_absの中でk番目に大きい値のインデックス
    # top_idx[k]の順位がkであることを確認
    for r, ti in enumerate(top_idx):
        # print(r, ti, vmap_diff_abs[ti])
        assert return_rank(vmap_diff_abs, ti) == r, f"Error: {ti}, {r}"
    places_to_fix = [[tgt_layer, pos] for pos in top_idx]
    # vmap_diff[:, tgt_layer]からconditionに合うものだけ取り出す
    tgt_vdiff = vmap_diff[top_idx, tgt_layer]
    return places_to_fix, tgt_vdiff

def localize_neurons_random(vscore_before_dir, vscore_dir, vscore_after_dir, tgt_layer, n, tgt_split="repair", misclf_pair=None, tgt_label=None, fpfn=None, rank_type="abs"):
    def _get_vscore_shape(vscore_dir):
        for cor_mis in ["cor", "mis"]:
            ds_type = f"ori_{tgt_split}"
            vscore_save_path = os.path.join(vscore_dir, f"vscore_l1tol12_all_label_{ds_type}_{cor_mis}.npy")
            if misclf_pair is not None and cor_mis == "mis":
                # If misclf_pair is specified, get only the data for that target
                assert len(misclf_pair) == 2, f"Error: {misclf_pair}"
                slabel, tlabel = misclf_pair
                vscore_save_path = os.path.join(vscore_dir, f"vscore_l1tol12_{slabel}to{tlabel}_{ds_type}_{cor_mis}.npy")
            if tgt_label is not None and cor_mis == "mis":
                # If tgt_label is specified, get only the data for that target
                vscore_save_path = os.path.join(vscore_dir, f"vscore_l1tol12_{tgt_label}_{ds_type}_{cor_mis}.npy")
                if fpfn is not None:
                    vscore_save_path = os.path.join(vscore_dir, f"vscore_l1tol12_{tgt_label}_{ds_type}_{fpfn}_{cor_mis}.npy")
            vscores = np.load(vscore_save_path)
            return vscores.shape
    vscore_shape = _get_vscore_shape(vscore_dir)
    num_neurons = vscore_shape[1]
    # ランダムにnum_neurons個のニューロンからn個のニューロンを選ぶ
    top_idx = np.random.choice(num_neurons, n, replace=False)
    places_to_fix = [[tgt_layer, pos] for pos in top_idx]
    return places_to_fix, None

def rank_descending(x):
    # x を降順に並べ替えるためのインデックスを取得
    sorted_indices = np.argsort(x)[::-1]
    # 順位用の空の配列を準備
    ranks = np.empty_like(sorted_indices)
    # インデックスを使って順位を設定
    ranks[sorted_indices] = np.arange(len(x))
    return ranks

def return_rank(x, i, order="desc"):
    # x[i] の順位を返す
    if order == "desc":
        return np.argsort(x)[::-1].tolist().index(i)
    elif order == "asc":
        return np.argsort(x).tolist().index(i)
    else:
        raise NotImplementedError

def localize_neurons_with_mean_activation(vscore_before_dir, vscore_dir, vscore_after_dir, tgt_layer, n, intermediate_states, tgt_mis_indices, tgt_split="repair", misclf_pair=None, tgt_label=None, fpfn=None, corruption_type=None, rank_type="abs", alpha=None, return_all_neuron_score=False, vscore_abs=False, covavg=True, vscore_cor_dir=None, return_before_norm=False):
    vmap_dic = defaultdict(np.array)
    # abs, covavg (vscore計算のバリエーション) によってファイル名が違う
    vscore_path_prefix = ("vscore_abs" if vscore_abs else "vscore") + ("_covavg" if covavg else "")
    for cor_mis in ["cor", "mis"]:
        ds_type = f"ori_{tgt_split}"
        # vscore_save_pathの設定
        vscore_save_path = os.path.join(vscore_dir, f"{vscore_path_prefix}_l1tol12_all_label_{ds_type}_{cor_mis}.npy")
        if cor_mis == "mis":
            if misclf_pair is not None:
                # If misclf_pair is specified, get only the data for that target
                assert len(misclf_pair) == 2, f"Error: {misclf_pair}"
                slabel, tlabel = misclf_pair
                vscore_save_path = os.path.join(vscore_dir, f"{vscore_path_prefix}_l1tol12_{slabel}to{tlabel}_{ds_type}_mis.npy")
            if tgt_label is not None:
                # If tgt_label is specified, get only the data for that target
                vscore_save_path = os.path.join(vscore_dir, f"{vscore_path_prefix}_l1tol12_{tgt_label}_{ds_type}_mis.npy")
                if fpfn is not None:
                    vscore_save_path = os.path.join(vscore_dir, f"{vscore_path_prefix}_l1tol12_{tgt_label}_{ds_type}_{fpfn}_mis.npy")
            if corruption_type is not None:
                # misclf_pair, tgt_label, fpfnが全部Noneであることを保証
                assert misclf_pair is None, f"Error: {misclf_pair}"
                assert tgt_label is None, f"Error: {tgt_label}"
                assert fpfn is None, f"Error: {fpfn}"
                vscore_save_path = os.path.join(vscore_dir, f"{vscore_path_prefix}_l1tol12_{tgt_label}_{corruption_type}_mis.npy")
        elif cor_mis == "cor":
            if misclf_pair is not None:
                label_str = f"label_{misclf_pair[1]}" # 正解ラベルに合わせたい
            else:
                assert tgt_label is not None, f"Error: {tgt_label}"
                if fpfn == "fn":
                    label_str = f"label_{tgt_label}" # みのがされたラベルに合わせたい
                else:
                    assert fpfn == "fp", f"Error: {fpfn}"
                    label_str = "all_label" # tgt_fpの時だけ正解ラベルがさまざまなので全体の正解を使う
            vscore_save_path = os.path.join(vscore_cor_dir, f"{vscore_path_prefix}_l1tol12_{label_str}_{ds_type}_cor.npy")
        # vscoreを読み込む
        print(f"vscore_save_path: {vscore_save_path}")
        vscores = np.load(vscore_save_path)
        vmap_dic[cor_mis] = vscores.T
    vmap_cor = vmap_dic["cor"]
    vmap_mis = vmap_dic["mis"]
    vmap_diff = vmap_cor - vmap_mis # shape: (num_neurons, num_layers) or (num_neurons,)
    # vmap_diffが2次元の場合は以下の処理
    if len(vmap_diff.shape) == 2:
        # vmap_diff[:, tgt_layer]の絶対値をvdiffに関するスコア
        vmap_diff_abs = np.abs(vmap_diff[:, tgt_layer]) # shape: (num_neurons,)
    elif len(vmap_diff.shape) == 1:
        vmap_diff_abs = np.abs(vmap_diff) # shape: (num_neurons,)
    # cache_statesから中間ニューロンの値を取得
    # print(intermediate_states.shape) # shape: (num_tgt_mis_samples, num_neurons)
    # 活性化後値の全対象誤分類サンプルにわたっての平均をmean_activationに関するスコア
    if tgt_mis_indices is None:
        # tgt_mis_indicesがNoneの場合は，全ての中間ニューロンの値を使用
        mean_activation = np.mean(intermediate_states, axis=0) # shape: (num_neurons,)
    else:
        mean_activation = np.mean(intermediate_states[tgt_mis_indices], axis=0) # shape: (num_neurons,)
    
    # min-max正規化する前の値を取っておく
    _vmap_diff_abs = vmap_diff_abs.copy()
    _mean_activation = mean_activation.copy()
    
    # vmap_diff_absとmean_activationをそれぞれmin-max正規化
    vmap_diff_abs = (vmap_diff_abs - np.min(vmap_diff_abs)) / (np.max(vmap_diff_abs) - np.min(vmap_diff_abs))
    mean_activation = (mean_activation - np.min(mean_activation)) / (np.max(mean_activation) - np.min(mean_activation))
    
    if alpha is None:
        # neuron_score として，上の2つのベクトルの要素ごとの積を使う
        neuron_score = vmap_diff_abs * mean_activation # shape: (num_neurons,)
    else:
        # neuron_score として，上の2つのベクトルの重み付き和
        neuron_score = alpha * vmap_diff_abs + (1-alpha) * mean_activation # shape: (num_neurons,)
    
    if n is not None:
        # neuron_scoreの上位n個を取得
        if isinstance(n, int):
            top_idx = np.argsort(neuron_score)[::-1][:n] # top_idx[k] = vmap_diff_absの中でk番目に大きい値のインデックス
        elif isinstance(n, float):
            assert n <= 1, f"Error: {n}"
            top_idx = np.argsort(neuron_score)[::-1][:int(len(neuron_score) * n)] # top_idx[k] = vmap_diff_absの中でk番目に大きい値のインデックス
        else:
            print(n)
            raise NotImplementedError
        # top_idx[k]の順位がkであることを確認
        for r, ti in enumerate(top_idx):
            # print(r, ti, vmap_diff_abs[ti])
            assert return_rank(neuron_score, ti) == r, f"Error: {ti}, {r}"
        places_to_fix = [[tgt_layer, pos] for pos in top_idx]
        # vmap_diff[:, tgt_layer]からconditionに合うものだけ取り出す
        tgt_neuron_score = neuron_score[top_idx]
    else: # n is Noneの場合は，降順ソートしてから全ニューロンの情報を返す
        top_idx = np.argsort(neuron_score)[::-1]
        # top_idx[k]の順位がkであることを確認
        for r, ti in enumerate(top_idx):
            # print(r, ti, vmap_diff_abs[ti])
            assert return_rank(neuron_score, ti) == r, f"Error: {ti}, {r}"
        places_to_fix = [[tgt_layer, pos] for pos in top_idx]
        tgt_neuron_score = neuron_score[top_idx]
    if not return_all_neuron_score:
        # top_idxのニューロンのスコアだけ返す
        return places_to_fix, tgt_neuron_score
    else:
        # 全ニューロンのスコアを返す
        if return_before_norm:
            return places_to_fix, tgt_neuron_score, neuron_score, _mean_activation, _vmap_diff_abs
        else:
            return places_to_fix, tgt_neuron_score, neuron_score

def localize_weights(vscore_before_dir, vscore_dir, vscore_after_dir, tgt_layer, n, tgt_split="repair", misclf_pair=None, tgt_label=None, fpfn=None, rank_type="abs"):

    vdiff_dic = defaultdict(defaultdict)
    # Iteration of neurons before intermediate neurons, intermediate neurons, and neurons after intermediate neurons
    for ba, vscore_dir in zip(["before", "intermediate", "after"], [vscore_before_dir, vscore_dir, vscore_after_dir]):
        vdiff_dic[ba] = defaultdict(np.array)
        vmap_dic = defaultdict(np.array)
        # Load vscore for correct and incorrect cases
        for cor_mis in ["cor", "mis"]:
            vmap_dic[cor_mis] = defaultdict(np.array)
            ds_type = f"ori_{tgt_split}"
            vscore_save_path = os.path.join(vscore_dir, f"vscore_l1tol12_all_label_{ds_type}_{cor_mis}.npy")
            if misclf_pair is not None and cor_mis == "mis":
                # If misclf_pair is specified, get only the data for that target
                assert len(misclf_pair) == 2, f"Error: {misclf_pair}"
                slabel, tlabel = misclf_pair
                vscore_save_path = os.path.join(vscore_dir, f"vscore_l1tol12_{slabel}to{tlabel}_{ds_type}_{cor_mis}.npy")
            if tgt_label is not None and cor_mis == "mis":
                # If tgt_label is specified, get only the data for that target
                vscore_save_path = os.path.join(vscore_dir, f"vscore_l1tol12_{tgt_label}_{ds_type}_{cor_mis}.npy")
                if fpfn is not None:
                    vscore_save_path = os.path.join(vscore_dir, f"vscore_l1tol12_{tgt_label}_{ds_type}_{fpfn}_{cor_mis}.npy")
            vscores = np.load(vscore_save_path)
            vmap_dic[cor_mis] = vscores.T
        vmap_cor = vmap_dic["cor"]
        vmap_mis = vmap_dic["mis"]
        # Save vdiff and its ranking in dictionary linked to ba
        vmap_diff = vmap_cor - vmap_mis
        if rank_type == "abs":
            vdiff_dic[ba]["vdiff"] = np.abs(vmap_diff[:, tgt_layer])
            vdiff_dic[ba]["rank"] = rank_descending(vdiff_dic[ba]["vdiff"]) # Descending order of absolute values
            order = "desc"
        else:
            vdiff_dic[ba]["vdiff"] = vmap_diff[:, tgt_layer]
            if rank_type == "desc":
                vdiff_dic[ba]["rank"] = rank_descending(vdiff_dic[ba]["vdiff"]) # Descending order of values before taking absolute value
                order = "desc"
            elif rank_type == "asc":
                vdiff_dic[ba]["rank"] = rank_descending(- vdiff_dic[ba]["vdiff"]) # Ascending order of values before taking absolute value
                order = "asc"
            else:
                raise NotImplementedError
        # Verify that the rank of the i-th value of vdiff_dic[ba]["vdiff"] equals the i-th value of vdiff_dic[ba]["rank"]
        for i, r in enumerate(vdiff_dic[ba]["rank"]):
            assert return_rank(vdiff_dic[ba]["vdiff"], i, order) == r, f"Error: {i}, {r}"
        # Print for verification of rank uniqueness
        # print(f"{len(set(vdiff_dic[ba]['rank']))} / {len(vdiff_dic[ba]['rank'])}")
        # print(f'({ba}) |vdiff| [min, max] = [{np.min(vdiff_dic[ba]["vdiff"])}, {np.max(vdiff_dic[ba]["vdiff"])}]')
    # Get top n from before,after and top 4n from intermediate
    top_idx_dic = defaultdict(list)
    for ba, dic in vdiff_dic.items():
        if ba == "intermediate":
            topx = 4*n
        else:
            topx = n
        top_idx_dic[ba] = np.where(dic["rank"] < topx)[0]
        # print(f"{ba}: {top_idx_dic[ba]}")
    # Return repair locations for before-intermediate, intermediate-after
    pos_before = np.array(list(product(top_idx_dic["intermediate"], top_idx_dic["before"])))
    pos_after = np.array(list(product(top_idx_dic["after"], top_idx_dic["intermediate"])))
    return pos_before, pos_after

def get_vscore_diff_and_sim(vscore_before_dir, vscore_dir, vscore_after_dir, tgt_split="repair", misclf_pair=None, tgt_label=None, fpfn=None):

    vdiff_dic = defaultdict(np.array)
    # Iteration of neurons before intermediate neurons, intermediate neurons, and neurons after intermediate neurons
    for ba, vscore_dir in zip(["before", "intermediate", "after"], [vscore_before_dir, vscore_dir, vscore_after_dir]):
        vdiff_dic[ba] = defaultdict(np.array)
        vmap_dic = defaultdict(np.array)
        # Load vscore for correct and incorrect cases
        for cor_mis in ["cor", "mis"]:
            vmap_dic[cor_mis] = defaultdict(np.array)
            ds_type = f"ori_{tgt_split}"
            vscore_save_path = os.path.join(vscore_dir, f"vscore_l1tol12_all_label_{ds_type}_{cor_mis}.npy")
            if misclf_pair is not None and cor_mis == "mis":
                # If misclf_pair is specified, get only the data for that target
                assert len(misclf_pair) == 2, f"Error: {misclf_pair}"
                slabel, tlabel = misclf_pair
                vscore_save_path = os.path.join(vscore_dir, f"vscore_l1tol12_{slabel}to{tlabel}_{ds_type}_{cor_mis}.npy")
            if tgt_label is not None and cor_mis == "mis":
                # If tgt_label is specified, get only the data for that target
                vscore_save_path = os.path.join(vscore_dir, f"vscore_l1tol12_{tgt_label}_{ds_type}_{cor_mis}.npy")
                if fpfn is not None:
                    vscore_save_path = os.path.join(vscore_dir, f"vscore_l1tol12_{tgt_label}_{ds_type}_{fpfn}_{cor_mis}.npy")
            print(f"vscore_save_path: {vscore_save_path}")
            vscores = np.load(vscore_save_path)
            vmap_dic[cor_mis] = vscores.T
        vmap_cor = vmap_dic["cor"]
        vmap_mis = vmap_dic["mis"]
        # Save vdiff and its ranking in dictionary linked to ba
        vmap_diff = vmap_cor - vmap_mis
        vdiff_dic[ba]["vdiff"] = vmap_diff
        dot_products = np.sum(vmap_cor * vmap_mis, axis=0)
        a_norms = np.linalg.norm(vmap_cor, axis=0)
        b_norms = np.linalg.norm(vmap_mis, axis=0)
        cosine_similarity = dot_products / (a_norms * b_norms)
        vdiff_dic[ba]["cos_sim"] = cosine_similarity
    return vdiff_dic

def localize_weights_random(vscore_before_dir, vscore_dir, vscore_after_dir, tgt_layer, n, tgt_split="repair", misclf_pair=None, tgt_label=None, fpfn=None, rank_type=None):
    def _get_vscore_shape(vscore_dir):
        for cor_mis in ["cor", "mis"]:
            ds_type = f"ori_{tgt_split}"
            vscore_save_path = os.path.join(vscore_dir, f"vscore_l1tol12_all_label_{ds_type}_{cor_mis}.npy")
            if misclf_pair is not None and cor_mis == "mis":
                # If misclf_pair is specified, get only the data for that target
                assert len(misclf_pair) == 2, f"Error: {misclf_pair}"
                slabel, tlabel = misclf_pair
                vscore_save_path = os.path.join(vscore_dir, f"vscore_l1tol12_{slabel}to{tlabel}_{ds_type}_{cor_mis}.npy")
            if tgt_label is not None and cor_mis == "mis":
                # If tgt_label is specified, get only the data for that target
                vscore_save_path = os.path.join(vscore_dir, f"vscore_l1tol12_{tgt_label}_{ds_type}_{cor_mis}.npy")
                if fpfn is not None:
                    vscore_save_path = os.path.join(vscore_dir, f"vscore_l1tol12_{tgt_label}_{ds_type}_{fpfn}_{cor_mis}.npy")
            vscores = np.load(vscore_save_path)
            return vscores.shape

    top_idx_dic = defaultdict(list)
    for ba, vscore_dir in zip(["before", "intermediate", "after"], [vscore_before_dir, vscore_dir, vscore_after_dir]):
        vscore_shape = _get_vscore_shape(vscore_dir)
        num_neurons = vscore_shape[1]
        # Randomly select n or 4n neurons
        if ba == "intermediate":
            topx = 4*n
        else:
            topx = n
        top_idx_dic[ba] = np.random.choice(num_neurons, topx, replace=False)
    # Return repair locations for before-intermediate, intermediate-after
    pos_before = np.array(list(product(top_idx_dic["intermediate"], top_idx_dic["before"])))
    pos_after = np.array(list(product(top_idx_dic["after"], top_idx_dic["intermediate"])))
    return pos_before, pos_after

class ViTFromLastLayer(nn.Module):
    def __init__(self, base_model):
        super(ViTFromLastLayer, self).__init__()
        self.base_model = base_model
        self.base_model.eval()
        self.base_model_last_layer = self.base_model.vit.encoder.layer[-1]

    def forward(self, hidden_states_before_layernorm, tgt_pos=None, tmp_score=None,
        imp_pos=None, imp_op=None):
        layer_output = self.base_model_last_layer.layernorm_after(hidden_states_before_layernorm)
        layer_output = self.base_model_last_layer.intermediate(layer_output, tgt_pos=tgt_pos, tmp_score=tmp_score, imp_pos=imp_pos, imp_op=imp_op)
        layer_output = self.base_model_last_layer.output(layer_output, hidden_states_before_layernorm)
        sequence_output = self.base_model.vit.layernorm(layer_output)
        logits = self.base_model.classifier(sequence_output[:, 0, :])
        return logits
    
# def generate_random_positions(start_layer_idx, end_layer_idx, num_neurons, num_kn):
#     """
#     Randomly select knowledge neuron positions (layer numbers >= start_layer_idx and <= end_layer_idx-1, neuron numbers >= 0 and <= num_neurons)
#     """
#     kn_list = []
#     for _ in range(num_kn):
#         layer_idx = np.random.randint(start_layer_idx, end_layer_idx)
#         neuron_idx = np.random.randint(num_neurons)
#         kn_list.append([layer_idx, neuron_idx])
#     return kn_list

def get_misclf_info(pred_labels, true_labels, num_classes):
    # Count number of misclassifications
    mis_matrix = np.zeros((num_classes, num_classes), dtype=int)
    mis_indices = {i: {j: [] for j in range(num_classes) if i != j} for i in range(num_classes)}
    for idx, (pred, true) in enumerate(zip(pred_labels, true_labels)):
        if pred != true:
            mis_matrix[pred, true] += 1
            mis_indices[pred][true].append(idx)  # Track the indices where the misclassification occurred
    # Create misclassification ranking
    mis_ranking = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j:
                mis_ranking.append((i, j, mis_matrix[i, j]))
    mis_ranking.sort(key=lambda x: x[2], reverse=True)
    print("Top 10 misclassification:")
    for i, j, mis in mis_ranking[:10]:
        print(f"pred {i} -> true {j}: {mis} / {mis_matrix.sum()} = {100 * mis / mis_matrix.sum():.2f} %")

    # Combine metrics for each class into one dictionary
    met_dict = defaultdict(np.array)
    precision, recall, f1_metric = evaluate.load("precision"), evaluate.load("recall"), evaluate.load("f1")
    precisions = precision.compute(predictions=pred_labels, references=true_labels, average=None) # Dictionary format {"metric_name": array of metric values for each class}
    recalls = recall.compute(predictions=pred_labels, references=true_labels, average=None)
    f1_scores = f1_metric.compute(predictions=pred_labels, references=true_labels, average=None)
    for met_item in [precisions, recalls, f1_scores]:
        met_dict.update(met_item)

    # Display pairs of idx and metric in order of worst metrics
    for metric in met_dict.keys():
        print(f"Top 10 worst {metric} scores:")
        met_ranking = sorted(enumerate(met_dict[metric]), key=lambda x: x[1])
        for idx, score in met_ranking[:10]:
            print(f"label: {idx}, {metric}: {score}")

    return mis_matrix, mis_ranking, mis_indices, met_dict

def src_tgt_selection(mis_ranking, mis_indices, tgt_rank):
    """
    Used when performing src-tgt type repair
    Extract the tgt_rank-th misclassification information from src-tgt type misclassifications.
    Specifically, extract prediction label, true label, and corresponding sample index.
    """
    # Extract target misclassification information from ranking
    slabel, tlabel, mis_cnt = mis_ranking[tgt_rank-1]
    tgt_mis_indices = mis_indices[slabel][tlabel]
    return slabel, tlabel, np.array(tgt_mis_indices)

def tgt_selection(met_dict, mis_indices, tgt_rank, used_met="f1"):
    """
    Used when performing tgt type repair
    In tgt type misclassifications, identify the label with the worst used_met at tgt_rank-th position and extract that information.
    Specifically, extract target label and corresponding sample index.
    """
    metrics = met_dict[used_met]
    num_labels = len(metrics)
    met_ranking = sorted(enumerate(metrics), key=lambda x: x[1])
    tgt_label = met_ranking[tgt_rank-1][0]
    tgt_mis_indices = []
    for pred_label in range(num_labels):
        for true_label in range(num_labels):
            # Change misclassification definition based on used_met
            if used_met == "f1": # false positive and false negative
                cond_fpfn = (pred_label == tgt_label or true_label == tgt_label)
            elif used_met == "precision": # false positive
                cond_fpfn = (pred_label == tgt_label)
            elif used_met == "recall": # false negative
                cond_fpfn = (true_label == tgt_label)
            # pred_label != true_label so it's a misclassified sample
            # pred_label == tgt_label is False positive, true_label == tgt_label is False negative
            if cond_fpfn and pred_label != true_label:
                tgt_mis_indices.extend(mis_indices[pred_label][true_label])
    return tgt_label, np.array(tgt_mis_indices)

def all_selection(mis_indices):
    """
    Used when performing all type repair.
    Flatten and concatenate all lists stored in mis_indices[i][j] into one dimension.
    """
    tgt_mis_indices = []
    for mi in mis_indices.values():
        for mij in mi.values():
            if len(mij) > 0:
                tgt_mis_indices.extend(mij)
    return np.array(tgt_mis_indices)

def identfy_tgt_misclf(misclf_info_dir, tgt_split="repair", misclf_type="tgt", tgt_rank=1, fpfn=None):
    # Load indices
    with open(os.path.join(misclf_info_dir, f"{tgt_split}_mis_indices.pkl"), "rb") as f:
        mis_indices = pickle.load(f)
    # Load ranking
    with open(os.path.join(misclf_info_dir, f"{tgt_split}_mis_ranking.pkl"), "rb") as f:
        mis_ranking = pickle.load(f)
    # Load metrics dict
    with open(os.path.join(misclf_info_dir, f"{tgt_split}_met_dict.pkl"), "rb") as f:
        met_dict = pickle.load(f)
    if misclf_type == "src_tgt":
        slabel, tlabel, tgt_mis_indices = src_tgt_selection(mis_ranking, mis_indices, tgt_rank)
        misclf_pair = (slabel, tlabel)
        tgt_label = None
    elif misclf_type == "tgt":
        if fpfn is None:
            used_met = "f1"
        elif fpfn == "fp":
            used_met = "precision"
        elif fpfn == "fn":
            used_met = "recall"
        else:
            NotImplementedError, f"fpfn: {fpfn}"
        tlabel, tgt_mis_indices = tgt_selection(met_dict, mis_indices, tgt_rank, used_met=used_met)
        tgt_label = tlabel
        misclf_pair = None
    elif misclf_type == "all":
        tgt_mis_indices = all_selection(mis_indices)
        tgt_label = None
        misclf_pair = None
    else:
        NotImplementedError, f"misclf_type: {misclf_type}"
    return misclf_pair, tgt_label, tgt_mis_indices

def get_ori_model_predictions(pred_res_dir, labels, tgt_split="repair", misclf_type="tgt", tgt_label=None):
    """
    Get prediction results for tgt_split of an old model that has already been saved as pkl.
    """
    # Get correct/incorrect indices for each sample in the repair set of the original model
    with open(os.path.join(pred_res_dir, f"{tgt_split}_pred.pkl"), "rb") as f:
        pred_res = pickle.load(f)
    pred_logits = pred_res.predictions
    ori_pred_labels = np.argmax(pred_logits, axis=-1)
    is_correct = ori_pred_labels == labels[tgt_split]
    indices_to_correct = np.where(is_correct)[0]
    if misclf_type == "tgt":
        assert tgt_label is not None, f"tgt_label should be specified when misclf_type is tgt."
        # If misclf_type == "tgt", treat only those that were correct with tgt_label as correct
        is_correct_tgt = is_correct & (labels[tgt_split] == tgt_label)
        indices_to_correct_tgt = np.where(is_correct_tgt)[0]
        # Set is_correct to the list excluding is_correct_for_tgt from is_correct
        is_correct_others = is_correct & (labels[tgt_split] != tgt_label)
        indices_to_correct_others = np.where(is_correct_others)[0]
        return ori_pred_labels, is_correct_tgt, indices_to_correct_tgt, is_correct_others, indices_to_correct_others
    return ori_pred_labels, is_correct, indices_to_correct

def get_new_model_predictions(vit_from_last_layer, batch_hs_before_layernorm, batch_labels, tgt_pos=0):
    """
    Get prediction results for tgt_split of a new model that has not been saved yet.
    """
    all_pred_labels = []
    all_true_labels = []
    for cache_state, y in zip(batch_hs_before_layernorm, batch_labels):
        logits = vit_from_last_layer(hidden_states_before_layernorm=cache_state, tgt_pos=tgt_pos)
        # Convert output logits to probabilities
        proba = torch.nn.functional.softmax(logits, dim=-1)
        pred_label = torch.argmax(proba, dim=-1)
        for pl, tl in zip(pred_label, y):
            all_pred_labels.append(pl.item())
            all_true_labels.append(tl)
    all_pred_labels = np.array(all_pred_labels)
    all_true_labels = np.array(all_true_labels)
    return all_pred_labels, all_true_labels

def get_batched_hs(hs_save_path, batch_size, tgt_indices=None, device=torch.device("cuda"), hs=None):
    if hs is None:
        hs_before_layernorm = torch.from_numpy(np.load(hs_save_path, mmap_mode="r")).to(device)
    else:
        hs_before_layernorm = hs
    if tgt_indices is not None:
        # Extract only the states for the indices to use
        hs_before_layernorm_tgt = hs_before_layernorm[tgt_indices]
    else:
        hs_before_layernorm_tgt = hs_before_layernorm
    num_batches = (hs_before_layernorm_tgt.shape[0] + batch_size - 1) // batch_size  # Calculate number of batches (round up to use the last incomplete batch)
    batch_hs_before_layernorm_tgt = np.array_split(hs_before_layernorm_tgt, num_batches)
    return batch_hs_before_layernorm_tgt

def get_batched_labels(labels, batch_size, tgt_indices=None):
    if tgt_indices is not None:
        labels_tgt = labels[tgt_indices]
    else:
        labels_tgt = labels
    num_batches = (len(labels_tgt) + batch_size - 1) // batch_size  # Calculate number of batches (round up to use the last incomplete batch)
    batch_labels = np.array_split(labels_tgt, num_batches)
    return batch_labels

def sample_from_correct_samples(num_sampled_from_correct, indices_to_correct):
    if num_sampled_from_correct < len(indices_to_correct):
        sampled_indices_to_correct = np.random.choice(indices_to_correct, num_sampled_from_correct, replace=False)
    else:
        sampled_indices_to_correct = indices_to_correct
    return sampled_indices_to_correct

# def sample_true_positive_indices_per_class(num_sampled_from_correct, indices_to_correct, ori_pred_labels):
#     # 予測ラベルごとにTrue Positiveのインデックスを取得
#     true_positive_indices_per_class = defaultdict(list)
#     for i, pred_label in enumerate(ori_pred_labels):
#         if i in indices_to_correct:
#             true_positive_indices_per_class[pred_label].append(i)
#     # true_positive_indices_per_classの各クラスのリストの長さを表示
#     tot = 0
#     for label, idx_list in true_positive_indices_per_class.items():
#         print(f"label: {label}, num_samples: {len(idx_list)}, num_sampled: {num_sampled_from_correct // len(idx_list)}")
#         tot += num_sampled_from_correct // len(idx_list)
#     print(f"total num_samples: {tot}")
#     exit()
#     # num_sampled_from_correctは合計のサンプル数なのでnum_classで割る
#     num_sampled_from_correct_per_class = num_sampled_from_correct // len(true_positive_indices_per_class.keys())
#     # 各クラスからnum_sampled_from_correct個ずつサンプリング
#     sampled_indices = []
#     for label, idx_list in true_positive_indices_per_class.items():
#         sampled_indices += np.random.choice(idx_list, num_sampled_from_correct_per_class, replace=False).tolist()
#     return np.array(sampled_indices)


def sample_true_positive_indices_per_class(
    num_sampled_from_correct,
    indices_to_correct,
    ori_pred_labels,
):
    """
    Randomly sample indices proportionally to class distribution from True Positive.

    Parameters
    ----------
    num_sampled_from_correct : int
        Total number of indices to extract (upper limit).
    indices_to_correct : Iterable[int]
        Set of indices of original data that are True Positive.
    ori_pred_labels : Sequence[int]
        Prediction label for each sample.

    Returns
    -------
    np.ndarray
        Extracted indices (dtype=int).
    """
    rng = np.random.default_rng()

    # --- 1) Collect TP indices for each class -----------------------------
    tp_per_class = defaultdict(list)
    indices_to_correct = set(indices_to_correct)  # Convert to set for O(1) reference
    for idx, pred_label in enumerate(ori_pred_labels):
        if idx in indices_to_correct:
            tp_per_class[pred_label].append(idx)

    if not tp_per_class:
        return np.array([], dtype=int)

    # --- 2) Allocate extraction numbers proportionally to class distribution -----------------------------------
    counts = {lbl: len(lst) for lbl, lst in tp_per_class.items()}
    total_tp = sum(counts.values())
    num_to_sample = min(num_sampled_from_correct, total_tp)
    print(f"sampling {num_to_sample} samples from {total_tp} samples...")

    # Initial allocation (round down with floor function)
    alloc = {lbl: (num_to_sample * cnt) // total_tp for lbl, cnt in counts.items()}
    # At this stage, the total allocation may be smaller than num_to_sample
    assert sum(alloc.values()) <= num_to_sample, f"alloc: {alloc}, num_to_sample: {num_to_sample}, total_tp: {total_tp}"

    # --- 3) Randomly allocate remainders to classes with capacity --------------------------
    remaining = num_to_sample - sum(alloc.values())
    leftover = {lbl: counts[lbl] - alloc[lbl] for lbl in counts}
    
    while remaining > 0:
        # Add classes that can still be extracted to candidates
        candidates = [lbl for lbl, cap in leftover.items() if cap > 0]
        if not candidates:
            break  # Just in case
        # Set probabilities proportional to remaining capacity
        probs = np.array([leftover[lbl] for lbl in candidates], dtype=float)
        probs /= probs.sum()
        chosen = rng.choice(candidates, p=probs)
        alloc[chosen] += 1
        leftover[chosen] -= 1
        remaining -= 1
    # At this stage, the total allocation should equal num_to_sample
    assert sum(alloc.values()) == num_to_sample, f"alloc: {alloc}, num_to_sample: {num_to_sample}, total_tp: {total_tp}"

    # --- 4) Sampling ----------------------------------------------------
    sampled = []
    for lbl, k in alloc.items():
        if k > 0:
            sampled.extend(rng.choice(tp_per_class[lbl], k, replace=False).tolist())
    assert len(sampled) == num_to_sample, f"len(sampled): {len(sampled)}, num_to_sample: {num_to_sample}, total_tp: {total_tp}"
    # Ensure that each index in sampled is included in the correct indices
    for idx in sampled:
        assert idx in indices_to_correct, f"Error: {idx} not in indices_to_correct"
    return np.array(sampled, dtype=int)

def maybe_initialize_repair_weights_(model, missing_keys):
    if any("intermediate.repair.weight" in key for key in missing_keys):
        print("🛠️ Initializing intermediate.repair.weight as identity matrix (for missing weights)")
        with torch.no_grad():
            for layer in model.vit.encoder.layer:
                W = layer.intermediate.repair.weight
                W.copy_(torch.eye(W.shape[0], device=W.device))
                W.requires_grad = False
    return model

class WeightedTrainer(Trainer):
    def __init__(self, *args, alpha=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha # Weight in loss calculation for correct/incorrect samples
        assert 0 <= alpha <= 1, f"alpha must be in [0, 1], but got {alpha}"
    def compute_loss(self, model, inputs, return_outputs=False):
        # NOTE: This function calculates loss for one batch
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss_per_sample = loss_fct(logits, labels) # (batch_size, )
        
        # 🔍 Calculate success/failure from prediction label vs true label
        pred_labels = torch.argmax(logits, dim=1)
        # Apply different filters to loss based on correct/incorrect for each batch
        is_correct = (pred_labels == labels).to(dtype=torch.float32)
        # Score definition
        score = torch.where(
            is_correct == 1,
            torch.ones_like(loss_per_sample),                      # 1 if correct
            1.0 / (loss_per_sample + 1.0)                 # 1 / (Loss + 1) if incorrect
        )
        # print(f"score: {score}")
        
        device = logits.device
        n_correct_batch = max((is_correct == 1).sum().item(), 1)
        n_incorrect_batch = max((is_correct == 0).sum().item(), 1)

        # Adjust weights using alpha
        sample_weights = torch.where(
            is_correct == 1,
            self.alpha / n_correct_batch,
            (1 - self.alpha) / n_incorrect_batch
        ).to(device)
        
        # print(f"weighted score: {score * sample_weights}")
        loss = - (score * sample_weights).sum()
        # print(f"loss: {loss}")

        return (loss, outputs) if return_outputs else loss