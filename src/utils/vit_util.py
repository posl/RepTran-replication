import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
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

def localize_neurons(vmap_dic, tgt_layer, theta=10):
    """
    vmap_dicのcorとmisの差分が上位theta%のニューロンを取得する

    Args:
        vmap_dic (dict): _description_
        tgt_layer (int): _description_
        theta (float, optional): _description_. Defaults to 10.
    """
    # take diff of vscores and get top 10% neurons
    vmap_cor = vmap_dic["cor"]
    vmap_mis = vmap_dic["mis"]
    vmap_diff = vmap_cor - vmap_mis # (num_neurons, num_layers)
    # vdiffの上位theta%のニューロンを取得
    theta = 10
    top_theta = np.percentile(np.abs(vmap_diff[:, tgt_layer]), 100-theta)
    condition = np.abs(vmap_diff[:, tgt_layer]).reshape(-1) > top_theta
    places_to_fix = [[tgt_layer, pos] for pos in np.where(condition)[0]]
    # vmap_diff[:, tgt_layer]からconditionに合うものだけ取り出す
    tgt_vdiff = vmap_diff[condition, tgt_layer]
    return places_to_fix, tgt_vdiff

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
    
def generate_random_positions(start_layer_idx, end_layer_idx, num_neurons, num_kn):
    """
    ランダムに知識ニューロンの位置 (start_layer_idx以上かつend_layer_idx-1以下のレイヤ番号, 0以上num_neurons以下のニューロン番号) を選ぶ
    """
    kn_list = []
    for _ in range(num_kn):
        layer_idx = np.random.randint(start_layer_idx, end_layer_idx)
        neuron_idx = np.random.randint(num_neurons)
        kn_list.append([layer_idx, neuron_idx])
    return kn_list