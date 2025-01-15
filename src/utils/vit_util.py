import os
import numpy as np
import pickle
import torch
import torch.nn as nn
from collections import defaultdict, Counter
from itertools import product
from transformers import ViTImageProcessor
import sys
sys.path.append('../')
from utils.constant import ViTExperiment
import evaluate

processor = ViTImageProcessor.from_pretrained(ViTExperiment.ViT_PATH)
met_acc = evaluate.load("accuracy")
met_f1 = evaluate.load("f1")

def transforms(batch):
    """
    画像のバッチを前処理する
    ラベルを表すカラム名がlabel (c10) の場合に適用可能
    
    Parameters
    ------------------
    
    Returns
    ------------------
    
    """
    # 画像のバッチを変換してtorch.tensorにする
    inputs = processor(images=batch["img"], return_tensors="pt")

    # ラベルのフィールドも前処理時に追加
    inputs["labels"] = batch["label"]
    return inputs

def transforms_c100(batch):
    """
    画像のバッチを前処理する
    ラベルを表すカラム名が fine_label (c100) の場合に適用可能
    
    Parameters
    ------------------
    
    Returns
    ------------------
    
    """
    # 画像のバッチを変換してtorch.tensorにする
    inputs = processor(images=batch["img"], return_tensors="pt")

    # ラベルのフィールドも前処理時に追加
    inputs["labels"] = batch["fine_label"]
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
        "f1": f1
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

def get_vscore(batch_neuron_values):
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
    # ニューロン分散共分散行列の対角成分 = 各ニューロンの分散 を取得
    neuron_var = np.diag(neuron_cov)
    # neuron_covの各行の和
    neuron_cov_sum = np.nansum(neuron_cov, axis=0) # 自分の分散 + (他の共分散の総和)
    # 他ニューロンとの共分散の平均
    mean_cov = (neuron_cov_sum - neuron_var) / (neuron_cov_sum.shape[0] - 1)
    # vscoreを計算
    vscore = neuron_var + mean_cov # (num_neurons_of_tgt_layer,)
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
                # misclf_pairが指定されている場合は，その対象のデータのみを取得
                assert len(misclf_pair) == 2, f"Error: {misclf_pair}"
                slabel, tlabel = misclf_pair
                vscore_save_path = os.path.join(vscore_dir, f"vscore_l1tol12_{slabel}to{tlabel}_{ds_type}_{cor_mis}.npy")
            if tgt_label is not None and cor_mis == "mis":
                # tgt_labelが指定されている場合は，その対象のデータのみを取得
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

def localize_neurons_with_mean_activation(vscore_before_dir, vscore_dir, vscore_after_dir, tgt_layer, n, intermediate_states, tgt_mis_indices, tgt_split="repair", misclf_pair=None, tgt_label=None, fpfn=None, rank_type="abs"):
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
    vmap_diff = vmap_cor - vmap_mis # shape: (num_neurons, num_layers)
    # vmap_diff[:, tgt_layer]の絶対値をvdiffに関するスコア
    vmap_diff_abs = np.abs(vmap_diff[:, tgt_layer]) # shape: (num_neurons, num_layers)
    
    # cache_statesから中間ニューロンの値を取得
    # print(intermediate_states.shape) # shape: (num_tgt_mis_samples, num_neurons)
    # 活性化後値の全対象誤分類サンプルにわたっての平均をmean_activationに関するスコア
    mean_activation = np.mean(intermediate_states[tgt_mis_indices], axis=0) # shape: (num_neurons,)
    
    # vmap_diff_absとmean_activationをそれぞれmin-max正規化
    vmap_diff_abs = (vmap_diff_abs - np.min(vmap_diff_abs)) / (np.max(vmap_diff_abs) - np.min(vmap_diff_abs))
    mean_activation = (mean_activation - np.min(mean_activation)) / (np.max(mean_activation) - np.min(mean_activation))
    
    # neuron_score として，上の2つのベクトルの要素ごとの積を使う
    neuron_score = vmap_diff_abs * mean_activation # shape: (num_neurons,)
    
    # neuron_scoreの上位n個を取得
    if isinstance(n, int):
        top_idx = np.argsort(neuron_score)[::-1][:n] # top_idx[k] = vmap_diff_absの中でk番目に大きい値のインデックス
    elif isinstance(n, float):
        assert n <= 1, f"Error: {n}"
        top_idx = np.argsort(neuron_score)[::-1][:int(len(neuron_score) * n)] # top_idx[k] = vmap_diff_absの中でk番目に大きい値のインデックス
    # top_idx[k]の順位がkであることを確認
    for r, ti in enumerate(top_idx):
        # print(r, ti, vmap_diff_abs[ti])
        assert return_rank(neuron_score, ti) == r, f"Error: {ti}, {r}"
    places_to_fix = [[tgt_layer, pos] for pos in top_idx]
    # vmap_diff[:, tgt_layer]からconditionに合うものだけ取り出す
    tgt_neuron_score = neuron_score[top_idx]
    return places_to_fix, tgt_neuron_score

def localize_weights(vscore_before_dir, vscore_dir, vscore_after_dir, tgt_layer, n, tgt_split="repair", misclf_pair=None, tgt_label=None, fpfn=None, rank_type="abs"):

    vdiff_dic = defaultdict(defaultdict)
    # 中間ニューロンの前のニューロン，中間ニューロン，中間ニューロンの後のニューロンそれぞれの繰り返し
    for ba, vscore_dir in zip(["before", "intermediate", "after"], [vscore_before_dir, vscore_dir, vscore_after_dir]):
        vdiff_dic[ba] = defaultdict(np.array)
        vmap_dic = defaultdict(np.array)
        # 正解と不正解時のvscoreを読み込む
        for cor_mis in ["cor", "mis"]:
            vmap_dic[cor_mis] = defaultdict(np.array)
            ds_type = f"ori_{tgt_split}"
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
            vscores = np.load(vscore_save_path)
            vmap_dic[cor_mis] = vscores.T
        vmap_cor = vmap_dic["cor"]
        vmap_mis = vmap_dic["mis"]
        # vdiffとそのランキングをbaに紐づいた辞書に保存
        vmap_diff = vmap_cor - vmap_mis
        if rank_type == "abs":
            vdiff_dic[ba]["vdiff"] = np.abs(vmap_diff[:, tgt_layer])
            vdiff_dic[ba]["rank"] = rank_descending(vdiff_dic[ba]["vdiff"]) # 絶対値の降順
            order = "desc"
        else:
            vdiff_dic[ba]["vdiff"] = vmap_diff[:, tgt_layer]
            if rank_type == "desc":
                vdiff_dic[ba]["rank"] = rank_descending(vdiff_dic[ba]["vdiff"]) # 絶対値取る前の値の大きい順
                order = "desc"
            elif rank_type == "asc":
                vdiff_dic[ba]["rank"] = rank_descending(- vdiff_dic[ba]["vdiff"]) # 絶対値取る前の値の小さい順
                order = "asc"
            else:
                raise NotImplementedError
        # vdiff_dic[ba]["vdiff"]のi番目の値の順位がvdiff_dic[ba]["rank"]のi番目と等しいことを確認
        for i, r in enumerate(vdiff_dic[ba]["rank"]):
            assert return_rank(vdiff_dic[ba]["vdiff"], i, order) == r, f"Error: {i}, {r}"
        # rankのユニーク性の確認のためのプリント
        # print(f"{len(set(vdiff_dic[ba]['rank']))} / {len(vdiff_dic[ba]['rank'])}")
        # print(f'({ba}) |vdiff| [min, max] = [{np.min(vdiff_dic[ba]["vdiff"])}, {np.max(vdiff_dic[ba]["vdiff"])}]')
    # before,afterからtop n個ずつ，intermediateからtop 4n個を取得
    top_idx_dic = defaultdict(list)
    for ba, dic in vdiff_dic.items():
        if ba == "intermediate":
            topx = 4*n
        else:
            topx = n
        top_idx_dic[ba] = np.where(dic["rank"] < topx)[0]
        # print(f"{ba}: {top_idx_dic[ba]}")
    # before-intermediate, intermediate-afterの修正箇所を返す
    pos_before = np.array(list(product(top_idx_dic["intermediate"], top_idx_dic["before"])))
    pos_after = np.array(list(product(top_idx_dic["after"], top_idx_dic["intermediate"])))
    return pos_before, pos_after

def get_vscore_diff_and_sim(vscore_before_dir, vscore_dir, vscore_after_dir, tgt_split="repair", misclf_pair=None, tgt_label=None, fpfn=None):

    vdiff_dic = defaultdict(np.array)
    # 中間ニューロンの前のニューロン，中間ニューロン，中間ニューロンの後のニューロンそれぞれの繰り返し
    for ba, vscore_dir in zip(["before", "intermediate", "after"], [vscore_before_dir, vscore_dir, vscore_after_dir]):
        vdiff_dic[ba] = defaultdict(np.array)
        vmap_dic = defaultdict(np.array)
        # 正解と不正解時のvscoreを読み込む
        for cor_mis in ["cor", "mis"]:
            vmap_dic[cor_mis] = defaultdict(np.array)
            ds_type = f"ori_{tgt_split}"
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
            print(f"vscore_save_path: {vscore_save_path}")
            vscores = np.load(vscore_save_path)
            vmap_dic[cor_mis] = vscores.T
        vmap_cor = vmap_dic["cor"]
        vmap_mis = vmap_dic["mis"]
        # vdiffとそのランキングをbaに紐づいた辞書に保存
        vmap_diff = vmap_cor - vmap_mis
        vdiff_dic[ba]["vdiff"] = vmap_diff
        dot_products = np.sum(vmap_cor * vmap_mis, axis=0)
        a_norms = np.linalg.norm(vmap_cor, axis=0)
        b_norms = np.linalg.norm(vmap_mis, axis=0)
        cosine_similarity = dot_products / (a_norms * b_norms)
        vdiff_dic[ba]["cos_sim"] = cosine_similarity
    return vdiff_dic

def localize_weights_random(vscore_before_dir, vscore_dir, vscore_after_dir, tgt_layer, n, tgt_split="repair", 
                            misclf_pair=None, tgt_label=None, fpfn=None, rank_type=None):
    def _get_vscore_shape(vscore_dir):
        for cor_mis in ["cor", "mis"]:
            ds_type = f"ori_{tgt_split}"
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
            vscores = np.load(vscore_save_path)
            return vscores.shape

    top_idx_dic = defaultdict(list)
    for ba, vscore_dir in zip(["before", "intermediate", "after"], [vscore_before_dir, vscore_dir, vscore_after_dir]):
        vscore_shape = _get_vscore_shape(vscore_dir)
        num_neurons = vscore_shape[1]
        # ランダムにn もしくは 4n個のニューロンを選ぶ
        if ba == "intermediate":
            topx = 4*n
        else:
            topx = n
        top_idx_dic[ba] = np.random.choice(num_neurons, topx, replace=False)
    # before-intermediate, intermediate-afterの修正箇所を返す
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
#     ランダムに知識ニューロンの位置 (start_layer_idx以上かつend_layer_idx-1以下のレイヤ番号, 0以上num_neurons以下のニューロン番号) を選ぶ
#     """
#     kn_list = []
#     for _ in range(num_kn):
#         layer_idx = np.random.randint(start_layer_idx, end_layer_idx)
#         neuron_idx = np.random.randint(num_neurons)
#         kn_list.append([layer_idx, neuron_idx])
#     return kn_list

def get_misclf_info(pred_labels, true_labels, num_classes):
    # 誤分類の数をカウント
    mis_matrix = np.zeros((num_classes, num_classes), dtype=int)
    mis_indices = {i: {j: [] for j in range(num_classes) if i != j} for i in range(num_classes)}
    for idx, (pred, true) in enumerate(zip(pred_labels, true_labels)):
        if pred != true:
            mis_matrix[pred, true] += 1
            mis_indices[pred][true].append(idx)  # Track the indices where the misclassification occurred
    # 誤分類のランキングを作成
    mis_ranking = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j:
                mis_ranking.append((i, j, mis_matrix[i, j]))
    mis_ranking.sort(key=lambda x: x[2], reverse=True)
    print("Top 10 misclassification:")
    for i, j, mis in mis_ranking[:10]:
        print(f"pred {i} -> true {j}: {mis} / {mis_matrix.sum()} = {100 * mis / mis_matrix.sum():.2f} %")

    # クラスごとのメトリクスを1つの辞書にまとめる
    met_dict = defaultdict(np.array)
    precision, recall, f1_metric = evaluate.load("precision"), evaluate.load("recall"), evaluate.load("f1")
    precisions = precision.compute(predictions=pred_labels, references=true_labels, average=None) # {"metric_name": クラスごとのmetric valueのarray} の辞書形式
    recalls = recall.compute(predictions=pred_labels, references=true_labels, average=None)
    f1_scores = f1_metric.compute(predictions=pred_labels, references=true_labels, average=None)
    for met_item in [precisions, recalls, f1_scores]:
        met_dict.update(met_item)

    # metricの悪い順にidxとmetricのペアを表示
    for metric in met_dict.keys():
        print(f"Top 10 worst {metric} scores:")
        met_ranking = sorted(enumerate(met_dict[metric]), key=lambda x: x[1])
        for idx, score in met_ranking[:10]:
            print(f"label: {idx}, {metric}: {score}")

    return mis_matrix, mis_ranking, mis_indices, met_dict

def src_tgt_selection(mis_ranking, mis_indices, tgt_rank):
    """
    src-tgt型のrepairをしたい場合に使う
    src-tgt型の誤分類において，tgt_rank番目の誤分類情報を取り出す．
    具体的には予測ラベル，正解ラベル，該当サンプルのインデックスを取り出す．
    """
    # ランキングから対象の誤分類情報を取り出す
    slabel, tlabel, mis_cnt = mis_ranking[tgt_rank-1]
    tgt_mis_indices = mis_indices[slabel][tlabel]
    return slabel, tlabel, np.array(tgt_mis_indices)

def tgt_selection(met_dict, mis_indices, tgt_rank, used_met="f1"):
    """
    tgt型のrepairをしたい場合に使う
    tgt型の誤分類において，tgt_rank番目にused_metの悪いラベルを特定し，その情報を取り出す．
    具体的には対象ラベル，該当サンプルのインデックスを取り出す．
    """
    metrics = met_dict[used_met]
    num_labels = len(metrics)
    met_ranking = sorted(enumerate(metrics), key=lambda x: x[1])
    tgt_label = met_ranking[tgt_rank-1][0]
    tgt_mis_indices = []
    for pred_label in range(num_labels):
        for true_label in range(num_labels):
            # used_metによって誤分類の定義を変える
            if used_met == "f1": # false positive and false negative
                cond_fpfn = (pred_label == tgt_label or true_label == tgt_label)
            elif used_met == "precision": # false positive
                cond_fpfn = (pred_label == tgt_label)
            elif used_met == "recall": # false negative
                cond_fpfn = (true_label == tgt_label)
            # pred_label != true_labelなので誤分類サンプル
            # pred_label == tgt_labelはFalse positive, true_label == tgt_labelはFalse negative
            if cond_fpfn and pred_label != true_label:
                tgt_mis_indices.extend(mis_indices[pred_label][true_label])
    return tgt_label, np.array(tgt_mis_indices)

def all_selection(mis_indices):
    """
    all型のrepairをする場合に使う.
    mis_indices[i][j]に格納されているすべてのリストを1次元にして結合する.
    """
    tgt_mis_indices = []
    for mi in mis_indices.values():
        for mij in mi.values():
            if len(mij) > 0:
                tgt_mis_indices.extend(mij)
    return np.array(tgt_mis_indices)

def identfy_tgt_misclf(misclf_info_dir, tgt_split="repair", misclf_type="tgt", tgt_rank=1, fpfn=None):
    # インデックスのロード
    with open(os.path.join(misclf_info_dir, f"{tgt_split}_mis_indices.pkl"), "rb") as f:
        mis_indices = pickle.load(f)
    # ランキングのロード
    with open(os.path.join(misclf_info_dir, f"{tgt_split}_mis_ranking.pkl"), "rb") as f:
        mis_ranking = pickle.load(f)
    # metrics dictのロード
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
    すでにpklで保存された, 古いモデルのtgt_splitに対する予測結果を取得する.
    """
    # original model の repair setの各サンプルに対する正解/不正解のインデックスを取得
    with open(os.path.join(pred_res_dir, f"{tgt_split}_pred.pkl"), "rb") as f:
        pred_res = pickle.load(f)
    pred_logits = pred_res.predictions
    ori_pred_labels = np.argmax(pred_logits, axis=-1)
    is_correct = ori_pred_labels == labels[tgt_split]
    indices_to_correct = np.where(is_correct)[0]
    if misclf_type == "tgt":
        assert tgt_label is not None, f"tgt_label should be specified when misclf_type is tgt."
        # misclf_type == "tgt"の場合は，tgt_labelで正解したものだけをcorrectとして扱う
        is_correct_tgt = is_correct & (labels[tgt_split] == tgt_label)
        indices_to_correct_tgt = np.where(is_correct_tgt)[0]
        # is_correctのリストからis_correct_for_tgtを除外したものをis_correctにする
        is_correct_others = is_correct & (labels[tgt_split] != tgt_label)
        indices_to_correct_others = np.where(is_correct_others)[0]
        return ori_pred_labels, is_correct_tgt, indices_to_correct_tgt, is_correct_others, indices_to_correct_others
    return ori_pred_labels, is_correct, indices_to_correct

def get_new_model_predictions(vit_from_last_layer, batch_hs_before_layernorm, batch_labels, tgt_pos=0):
    """
    まだ保存されていない, 新しいモデルのtgt_splitに対する予測結果を取得する.
    """
    all_pred_labels = []
    all_true_labels = []
    for cache_state, y in zip(batch_hs_before_layernorm, batch_labels):
        logits = vit_from_last_layer(hidden_states_before_layernorm=cache_state, tgt_pos=tgt_pos)
        # 出力されたlogitsを確率に変換
        proba = torch.nn.functional.softmax(logits, dim=-1)
        pred_label = torch.argmax(proba, dim=-1)
        for pl, tl in zip(pred_label, y):
            all_pred_labels.append(pl.item())
            all_true_labels.append(tl)
    all_pred_labels = np.array(all_pred_labels)
    all_true_labels = np.array(all_true_labels)
    return all_pred_labels, all_true_labels

def get_batched_hs(hs_save_path, batch_size, tgt_indices=None, device=torch.device("cuda")):
    hs_before_layernorm = torch.from_numpy(np.load(hs_save_path)).to(device)
    if tgt_indices is not None:
        # 使うインデックスに対する状態だけを取り出す
        hs_before_layernorm_tgt = hs_before_layernorm[tgt_indices]
    else:
        hs_before_layernorm_tgt = hs_before_layernorm
    num_batches = (hs_before_layernorm_tgt.shape[0] + batch_size - 1) // batch_size  # バッチの数を計算 (最後の中途半端なバッチも使いたいので，切り上げ)
    batch_hs_before_layernorm_tgt = np.array_split(hs_before_layernorm_tgt, num_batches)
    return batch_hs_before_layernorm_tgt

def get_batched_labels(labels, batch_size, tgt_indices=None):
    if tgt_indices is not None:
        labels_tgt = labels[tgt_indices]
    else:
        labels_tgt = labels
    num_batches = (len(labels_tgt) + batch_size - 1) // batch_size  # バッチの数を計算 (最後の中途半端なバッチも使いたいので，切り上げ)
    batch_labels = np.array_split(labels_tgt, num_batches)
    return batch_labels

def sample_from_correct_samples(num_sampled_from_correct, indices_to_correct):
    if num_sampled_from_correct < len(indices_to_correct):
        sampled_indices_to_correct = np.random.choice(indices_to_correct, num_sampled_from_correct, replace=False)
    else:
        sampled_indices_to_correct = indices_to_correct
    return sampled_indices_to_correct

def sample_true_positive_indices_per_class(num_sampled_from_correct, indices_to_correct, ori_pred_labels):
    # 予測ラベルごとにTrue Positiveのインデックスを取得
    true_positive_indices_per_class = defaultdict(list)
    for i, pred_label in enumerate(ori_pred_labels):
        if i in indices_to_correct:
            true_positive_indices_per_class[pred_label].append(i)
    # num_sampled_from_correctは合計のサンプル数なのでnum_classで割る
    num_sampled_from_correct_per_class = num_sampled_from_correct // len(true_positive_indices_per_class.keys())
    # 各クラスからnum_sampled_from_correct個ずつサンプリング
    sampled_indices = []
    for label, idx_list in true_positive_indices_per_class.items():
        sampled_indices += np.random.choice(idx_list, num_sampled_from_correct_per_class, replace=False).tolist()
    return np.array(sampled_indices)