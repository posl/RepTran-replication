import os, sys, math
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import argparse
import torch
import pickle
import evaluate
met_acc = evaluate.load("accuracy")
met_f1 = evaluate.load("f1")
from datasets import load_from_disk
from transformers import DefaultDataCollator, ViTForImageClassification, Trainer
from utils.helper import get_device
from utils.vit_util import processor, transforms, compute_metrics, transforms_c100, localize_weights
from utils.constant import ViTExperiment
from utils.log import set_exp_logging
from logging import getLogger

logger = getLogger("base_logger")

n=10 # TODO: 絶対ダメ

if __name__ == "__main__":
    # データセットをargparseで受け取る
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str)
    parser.add_argument('k', type=int, help="the fold id (0 to K-1)")
    parser.add_argument('patch_filename', type=str, help="the filename of the patch to be evaluated")
    args = parser.parse_args()
    ds_name = args.ds
    k = args.k
    patch_filename = args.patch_filename

    # デバイス (cuda, or cpu) の取得
    device = get_device()
    # datasetをロード (初回の読み込みだけやや時間かかる)
    dataset_dir = ViTExperiment.DATASET_DIR
    # バッチごとの処理のためのdata_collator
    data_collator = DefaultDataCollator()
    # オリジナルのds name
    ds_ori_name = ds_name.rstrip("c") if ds_name.endswith("c") else ds_name
    # foldが記載されたdir name
    ds_dirname = f"{ds_ori_name}_fold{k}"
    # dsのload
    ds = load_from_disk(os.path.join(dataset_dir, ds_dirname))
    ds_preprocessed = ds.with_transform(transforms)

    # datasetごとに違う変数のセット
    if ds_name == "c10" or ds_name == "c10c":
        tf_func = transforms
        label_col = "label"
    elif ds_name == "c100" or ds_name == "c100c":
        tf_func = transforms_c100
        label_col = "fine_label"
    else:
        NotImplementedError
    # 読み込まれた時にリアルタイムで前処理を適用するようにする
    ds_preprocessed = ds.with_transform(tf_func)
    # ラベルを示す文字列のlist
    labels = ds_preprocessed["train"].features[label_col].names
    
    # pretrained modelのロード
    ori_pretrained_dir = getattr(ViTExperiment, ds_ori_name).OUTPUT_DIR.format(k=k)
    pretrained_dir = os.path.join(ori_pretrained_dir, "repair_weight_by_de")
    # このpythonのファイル名を取得
    this_file_name = os.path.basename(__file__).split(".")[0]
    logger = set_exp_logging(exp_dir=pretrained_dir, exp_name=this_file_name)
    logger.info(f"ds_name: {ds_name}, fold_id: {k}")

    model = ViTForImageClassification.from_pretrained(ori_pretrained_dir).to(device)
    model.eval()
    end_li = model.vit.config.num_hidden_layers
    # 学習時の設定をロード
    training_args = torch.load(os.path.join(ori_pretrained_dir, "training_args.bin"))
    # Trainerオブジェクトの作成
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        train_dataset=ds_preprocessed["train"],
        eval_dataset=ds_preprocessed["test"],
        tokenizer=processor,
    )
    training_args_dict = training_args.to_dict()
    train_batch_size = training_args_dict["per_device_train_batch_size"]
    eval_batch_size = training_args_dict["per_device_eval_batch_size"]
    # 予測結果格納ディレクトリ
    pred_out_dir = os.path.join(pretrained_dir, "pred_results", os.path.splitext(patch_filename)[0])
    # pred_out_dirがなければ作成
    if not os.path.exists(pred_out_dir):
        os.makedirs(pred_out_dir)
    tgt_layer = 11
    tgt_pos = ViTExperiment.CLS_IDX
    # deの出力したパッチをロード
    best_patch = np.load(os.path.join(pretrained_dir, patch_filename))
    # 修復対象の位置をロード
    vscore_before_dir = os.path.join(ori_pretrained_dir, "vscores_before")
    vscore_dir = os.path.join(ori_pretrained_dir, "vscores")
    vscore_after_dir = os.path.join(ori_pretrained_dir, "vscores_after")
    pos_before, pos_after = localize_weights(vscore_before_dir, vscore_dir, vscore_after_dir, tgt_layer, n)
    logger.info(f"pos_before={pos_before}")
    logger.info(f"pos_after={pos_after}")
    logger.info(f"num(pos_to_fix)=num(pos_before)+num(pos_before)={len(pos_before)}+{len(pos_after)}={len(pos_before)+len(pos_after)}")
    logger.info(f"best_patch={best_patch}")
    logger.info(f"best_patch.shape: {best_patch.shape}")

    # ===============================================
    # patch setting
    # ===============================================

    for ba, pos in enumerate([pos_before, pos_after]):
        # patch_candidateのindexを設定
        if ba == 0:
            idx_patch_candidate = range(0, len(pos_before))
            tgt_weight_data = model.vit.encoder.layer[tgt_layer].intermediate.dense.weight.data
        else:
            idx_patch_candidate = range(len(pos_before), len(pos_before) + len(pos_after))
            tgt_weight_data = model.vit.encoder.layer[tgt_layer].output.dense.weight.data
        # posで指定された位置のニューロンを書き換える
        xi, yi = pos[:, 0], pos[:, 1]
        tgt_weight_data[xi, yi] = torch.from_numpy(best_patch[idx_patch_candidate]).to(device)

    # C10 or C100データセットに対する推論
    # ====================================================================
    if ds_name == "c10" or ds_name == "c100":
        # データセットのサイズとバッチサイズからイテレーション回数を計算
        train_iter = math.ceil(len(ds_preprocessed["train"]) / train_batch_size)
        repair_iter = math.ceil(len(ds_preprocessed["repair"]) / eval_batch_size)
        test_iter = math.ceil(len(ds_preprocessed["test"]) / eval_batch_size)
        # 各splitに対して予測を実行
        for split, batch_size in zip(["train", "repair", "test"], [train_batch_size, eval_batch_size, eval_batch_size]):
            logger.info(f"Process {split} (batch_size={batch_size})...")
            all_proba = []
            all_pred_labels = []
            # 各バッチに対する予測を実行
            for entry_dic in tqdm(ds_preprocessed[split].iter(batch_size=batch_size), total=len(ds_preprocessed[split])//batch_size+1):
                x, y = entry_dic["pixel_values"].to(device), entry_dic["labels"]
                output = model.forward(x, tgt_pos=tgt_pos, output_hidden_states_before_layernorm=False, output_intermediate_states=False)
                proba = torch.nn.functional.softmax(output.logits, dim=-1).cpu().detach().numpy()
                # 予測結果のラベルを取得
                pred_labels = proba.argmax(axis=-1)
                all_proba.append(proba)
                all_pred_labels.append(pred_labels)
            all_proba = np.concatenate(all_proba)
            all_pred_labels = np.concatenate(all_pred_labels)
            true_labels = np.array(ds[split][label_col])
            logger.info(f"all_proba.shape: {all_proba.shape}, all_pred_labels.shape: {all_pred_labels.shape}, true_labels.shape: {true_labels.shape}")
            # all_proba, all_pred_labels, true_labelsをnpzで保存
            np.savez(os.path.join(pred_out_dir, f"{split}_pred_results.npz"), all_proba=all_proba, all_pred_labels=all_pred_labels, true_labels=true_labels)
            logger.info(f"{split}_pred_results.npz saved")
            # 正解ラベルごとのpred_probaを保存
            os.makedirs(os.path.join(pred_out_dir, "class_proba"), exist_ok=True)
            for c in set(true_labels.tolist()):
                tgt_proba = all_proba[true_labels == c]
                np.save(os.path.join(pred_out_dir, "class_proba", f"{split}_proba_{c}.npy"), tgt_proba)
                logger.info(f"{split}_proba_{c}.npy ({tgt_proba.shape}) saved")
            acc = met_acc.compute(references=true_labels, predictions=all_pred_labels)
            f1 = met_f1.compute(references=true_labels, predictions=all_pred_labels, average="macro")
            logger.info(f"metrics for {split} set: acc={acc}, f1={f1}")
    else:
        # アラートを表示しつつNotImplementedErrorをraise
        print(f"ds_name: {ds_name} is not implemented yet")
        raise NotImplementedError