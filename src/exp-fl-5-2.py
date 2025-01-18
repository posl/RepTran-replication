import os, sys, time, pickle, json, math
import shutil
import torch
from tqdm import tqdm
from collections import defaultdict
from itertools import product
import numpy as np
import pandas as pd
from utils.helper import get_device, json2dict
from utils.vit_util import transforms_c100, compute_metrics, processor, get_misclf_info
from utils.constant import ViTExperiment, Experiment3
from utils.log import set_exp_logging
from logging import getLogger
from datasets import load_from_disk
from transformers import ViTForImageClassification, DefaultDataCollator, Trainer
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")

logger = getLogger("base_logger")
tgt_pos = ViTExperiment.CLS_IDX
exp_dir = "./exp-fl-5"
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

device = get_device()
ds_name = "c100"
num_fold = 5
tgt_layer = 11 # ViT-baseの最終層
tgt_split = "repair"

for k in range(num_fold):
    print(f"ds_name: {ds_name}, fold_id: {k}")
    save_dir = os.path.join(exp_dir, f"{ds_name}_fold{k}")
    os.makedirs(save_dir, exist_ok=True)
    
    if ds_name == "c100":
        num_classes = 100
        tf_func = transforms_c100
        label_col = "fine_label"
    else:
        raise NotImplementedError(f"ds_name: {ds_name}")

    # datasetをロード (true_labelsが欲しいので)
    ds_dirname = f"{ds_name}_fold{k}"
    ds = load_from_disk(os.path.join(ViTExperiment.DATASET_DIR, ds_dirname))
    data_collator = DefaultDataCollator()
    ds_preprocessed = ds.with_transform(tf_func)
    
    # pretrained modelのディレクトリ
    pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR.format(k=k)
    training_args = torch.load(os.path.join(pretrained_dir, "training_args.bin"))
    # 1エポック目のモデルロード
    model = ViTForImageClassification.from_pretrained(os.path.join(pretrained_dir, "checkpoint-1250"))
    model.to(device)
    
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
    
    # tgt_splitに対する推論を実行
    num_iter = math.ceil(len(ds_preprocessed[tgt_split]) / eval_batch_size)
    print(f"predict {tgt_split} data... #iter = {num_iter} ({len(ds_preprocessed[tgt_split])} samples / {eval_batch_size} batches)")
    pred_out = trainer.predict(ds_preprocessed[tgt_split])
    
    # 予測ラベルと正解ラベルからmisclassification情報を取得
    pred_labels = pred_out.predictions.argmax(-1)
    true_labels = pred_out.label_ids
    # 気持ち程度にaccとf1だしとく
    met_dict = compute_metrics(pred_out)
    print(f"Accuracy: {met_dict['eval_acc']}, F1: {met_dict['eval_f1']}")
    # 予測と正解ラベルを保存
    np.save(os.path.join(save_dir, f"{tgt_split}_true_labels.npy"), true_labels)
    np.save(os.path.join(save_dir, f"{tgt_split}_pred_labels.npy"), pred_labels)
    print(f"Saved: {tgt_split}_true_labels.npy, shape: {true_labels.shape}")
    print(f"Saved: {tgt_split}_pred_labels.npy, shape: {pred_labels.shape}")
    # misclassification情報を取得
    mis_matrix, mis_ranking, mis_indices, met_dict = get_misclf_info(pred_labels, true_labels, num_classes)
    
    # mis_matrixはnpyで，それ以外はpklで保存
    np.save(os.path.join(save_dir, f"{tgt_split}_mis_matrix.npy"), mis_matrix)
    with open(os.path.join(save_dir, f"{tgt_split}_mis_ranking.pkl"), "wb") as f:
        pickle.dump(mis_ranking, f)
    with open(os.path.join(save_dir, f"{tgt_split}_mis_indices.pkl"), "wb") as f:
        pickle.dump(mis_indices, f)
    with open(os.path.join(save_dir, f"{tgt_split}_met_dict.pkl"), "wb") as f:
        pickle.dump(met_dict, f)
    print("Summary of the misclassification info:")
    print(f"mis_matrix: {mis_matrix.shape}")
    print(f"total_mis: {mis_matrix.sum()}")