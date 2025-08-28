"""
We use the whole repair set in 002a,
but we use the part of repair set used for the repair in 002c.
"""

import os
import torch
import numpy as np
import argparse
from datasets import load_from_disk
from transformers import DefaultDataCollator, ViTForImageClassification, Trainer
from utils.helper import get_device, get_corruption_types
from utils.vit_util import processor, transforms, transforms_c100, compute_metrics, identfy_tgt_misclf, get_ori_model_predictions, sample_from_correct_samples, sample_true_positive_indices_per_class
from utils.constant import ViTExperiment

def retraining_with_repair_set(ds_name, k, tgt_rank, misclf_type, tgt_split, fpfn, num_sampled_from_correct=200, include_other_TP_for_fitness=True):
    # Set different variables for each dataset
    if ds_name == "c10":
        tf_func = transforms
        label_col = "label"
    elif ds_name == "c100":
        tf_func = transforms_c100
        label_col = "fine_label"
    else:
        NotImplementedError

    # Get device (cuda or cpu)
    device = get_device()
    dataset_dir = ViTExperiment.DATASET_DIR
    data_collator = DefaultDataCollator()
    ds_dirname = f"{ds_name}_fold{k}"

    # 対象のcorruption, severityのdatasetをロード
    ds = load_from_disk(os.path.join(dataset_dir, ds_dirname))
    # Apply preprocessing in real-time when loaded
    ds_preprocessed = ds.with_transform(tf_func)
    ds_train, ds_repair, ds_test = ds_preprocessed["train"], ds_preprocessed["repair"], ds_preprocessed["test"]

    labels = {
        "train": np.array(ds["train"][label_col]),
        "repair": np.array(ds["repair"][label_col]),
        "test": np.array(ds["test"][label_col])
    }

    # Load pretrained model
    pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR.format(k=k)
    # Extract misclassification information for tgt_rank
    misclf_info_dir = os.path.join(pretrained_dir, "misclf_info")
    misclf_pair, tgt_label, tgt_mis_indices = identfy_tgt_misclf(misclf_info_dir, tgt_split=tgt_split, tgt_rank=tgt_rank, misclf_type=misclf_type, fpfn=fpfn)
    indices_to_incorrect = tgt_mis_indices
    # NOTE: No need to sample from indices_to_incorrect? Currently using all, so randomness only comes from correct sampling
    # Get correct/incorrect indices for each sample in the repair set of the original model
    pred_res_dir = os.path.join(pretrained_dir, "pred_results", "PredictionOutput")
    if misclf_type == "tgt":
        ori_pred_labels, is_correct, indices_to_correct, is_correct_others, indices_to_correct_others = get_ori_model_predictions(pred_res_dir, labels, tgt_split=tgt_split, misclf_type=misclf_type, tgt_label=tgt_label)
    else:
        ori_pred_labels, is_correct, indices_to_correct = get_ori_model_predictions(pred_res_dir, labels, tgt_split=tgt_split, misclf_type=misclf_type, tgt_label=tgt_label)

    # Randomly sample a certain number from correct data for repair
    sampled_indices_to_correct = sample_from_correct_samples(num_sampled_from_correct, indices_to_correct)
    if include_other_TP_for_fitness and misclf_type == "tgt":
        for pl, tl in zip(ori_pred_labels[indices_to_correct_others], labels[tgt_split][indices_to_correct_others]):
            assert pl != tgt_label, f"pl: {pl}, tgt_label: {tgt_label}"
            assert tl != tgt_label, f"tl: {tl}, tgt_label: {tgt_label}"
            assert pl == tl, f"pl: {pl}, tl: {tl}"
        other_TP_indices = sample_true_positive_indices_per_class(num_sampled_from_correct, indices_to_correct_others, ori_pred_labels)
        sampled_indices_to_correct = np.concatenate([sampled_indices_to_correct, other_TP_indices])
    print(f"len(indices_to_correct): {len(sampled_indices_to_correct)}, len(indices_to_incorrect): {len(indices_to_incorrect)}")
    # Combine extracted correct data and all incorrect data into one dataset
    tgt_indices = sampled_indices_to_correct.tolist() + indices_to_incorrect.tolist() # .tolist() is a non-destructive method
    print(f"len(tgt_indices): {len(tgt_indices)}")
    ds_tgt = ds_repair.select(indices=tgt_indices)
    model = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
    model.eval()
    # オリジナルモデルのLoad training configuration
    training_args = torch.load(os.path.join(pretrained_dir, "training_args.bin"))
    # オリジナルの学習から設定を少し変える
    if fpfn is None:
        adapt_out_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_retraining_with_only_repair_target")
    else:
        adapt_out_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_{fpfn}_retraining_with_only_repair_target")
    os.makedirs(adapt_out_dir, exist_ok=True)
    training_args.output_dir = adapt_out_dir
    # print(training_args.num_epochs)
    if misclf_type == "src_tgt" or misclf_type == "tgt":
        training_args.num_train_epochs = 1 # NOTE: これは対象誤分類のデータの数によるが, src_tgtの場合はデータが少ないので1epochで十分
    else:
        training_args.num_train_epochs = 2
    training_args.logging_strategy = "no"
    # Create Trainer object
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        train_dataset=ds_tgt, #NOTE: repair setのうちrepairに使ったインデックスだけ取り出したサブセットをtrainingに使う
        eval_dataset=ds_test,
        tokenizer=processor,
    )
    # エポック数を表示
    # Execute training
    train_results = trainer.train()
    # Save
    trainer.save_model() # Can be loaded with from_pretrained()
    trainer.save_state() # save_model()よりも多くの情報をSaveする

if __name__ == "__main__":
    # Accept dataset via argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str, help="dataset name")
    parser.add_argument('k', type=int, help="the fold id (0 to K-1)")
    parser.add_argument('--tgt_rank', type=int, help="the rank of the target misclassification type", default=1)
    parser.add_argument('--misclf_type', type=str, help="the type of misclassification (src_tgt or tgt or all)", default="src_tgt")
    parser.add_argument("--fpfn", type=str, help="the type of misclassification (fp or fn)", default=None, choices=["fp", "fn"])
    args = parser.parse_args()
    ds_name = args.ds
    k = args.k
    tgt_rank = args.tgt_rank
    fpfn = args.fpfn
    misclf_type = args.misclf_type
    print(f"ds_name: {ds_name}, fold_id: {k}")
    retraining_with_repair_set(ds_name, k, tgt_rank, misclf_type, tgt_split="repair", fpfn=fpfn)