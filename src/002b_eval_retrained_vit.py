import os, sys, math
import numpy as np
import argparse
import torch
import pickle
from datasets import load_from_disk
from transformers import DefaultDataCollator, ViTForImageClassification, Trainer
from utils.helper import get_device
from utils.vit_util import processor, transforms, compute_metrics, transforms_c100, pred_to_proba
from utils.constant import ViTExperiment
from utils.log import set_exp_logging
from logging import getLogger

logger = getLogger("base_logger")

if __name__ == "__main__":
    # Accept dataset via argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str)
    parser.add_argument('k', type=int, help="the fold id (0 to K-1)")
    parser.add_argument('misclf_type', type=str, help="the type of misclassification (src_tgt or tgt or all)")
    parser.add_argument("--use_whole", action="store_true", help="use the whole dataset for evaluation")
    parser.add_argument('--tgt_rank', type=int, help="the rank of the target misclassification type", default=None)
    parser.add_argument("--fpfn", type=str, help="the type of misclassification (fp or fn)", default=None, choices=["fp", "fn"])
    args = parser.parse_args()
    ds_name = args.ds
    k = args.k
    misclf_type = args.misclf_type
    tgt_rank = args.tgt_rank
    use_whole = args.use_whole
    fpfn = args.fpfn
    # use_wholeがFalseの時はtgt_rankが指定されてないといけない
    if not use_whole:
        assert tgt_rank is not None, "tgt_rank is required when use_whole is False."
    print(f"ds_name: {ds_name}, fold_id: {k}, use_whole: {use_whole}, tgt_rank: {tgt_rank}, misclf_type: {misclf_type}, fpfn: {fpfn}")
    
    # Get device (cuda or cpu)
    device = get_device()
    # Load dataset (takes some time only on first load)
    dataset_dir = ViTExperiment.DATASET_DIR
    # data_collator for batch processing
    data_collator = DefaultDataCollator()
    # オリジナルのds name
    ds_ori_name = ds_name.rstrip("c") if ds_name.endswith("c") else ds_name
    # foldが記載されたdir name
    ds_dirname = f"{ds_ori_name}_fold{k}"
    # dsのload
    ds = load_from_disk(os.path.join(dataset_dir, ds_dirname))
    ds_preprocessed = ds.with_transform(transforms)
    
    # Set different variables for each dataset
    if ds_name == "c10" or ds_name == "c10c":
        tf_func = transforms
        label_col = "label"
    elif ds_name == "c100" or ds_name == "c100c":
        tf_func = transforms_c100
        label_col = "fine_label"
    else:
        NotImplementedError
    # Apply preprocessing in real-time when loaded
    ds_preprocessed = ds.with_transform(tf_func)
    # List of strings representing labels
    labels = ds_preprocessed["train"].features[label_col].names
    
    # Load pretrained model
    ori_pretrained_dir = getattr(ViTExperiment, ds_ori_name).OUTPUT_DIR.format(k=k)
    if use_whole:
        pretrained_dir = os.path.join(ori_pretrained_dir, "retraining_with_repair_set")
    elif fpfn is not None:
        pretrained_dir = os.path.join(ori_pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_{fpfn}_retraining_with_only_repair_target")
    else:
        pretrained_dir = os.path.join(ori_pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_retraining_with_only_repair_target")
    print(f"retrained model dir: {pretrained_dir}")
    # Get this Python file name
    this_file_name = os.path.basename(__file__).split(".")[0]
    logger = set_exp_logging(exp_dir=pretrained_dir, exp_name=this_file_name)

    model = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
    model.eval()
    # Load training configuration
    training_args = torch.load(os.path.join(pretrained_dir, "training_args.bin"))
    # Create Trainer object
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
    # Directory for storing prediction results
    pred_out_dir = os.path.join(pretrained_dir, "pred_results", "PredictionOutput")
    # Create pred_out_dir if it doesn't exist
    if not os.path.exists(pred_out_dir):
        os.makedirs(pred_out_dir)

    # C10 or C100データセットに対する推論
    # ====================================================================
    if ds_name == "c10" or ds_name == "c100":
        # Calculate number of iterations from dataset size and batch size
        train_iter = math.ceil(len(ds_preprocessed["train"]) / train_batch_size)
        repair_iter = math.ceil(len(ds_preprocessed["repair"]) / eval_batch_size)
        test_iter = math.ceil(len(ds_preprocessed["test"]) / eval_batch_size)
        # Execute inference on train, repair, test data
        # train
        print(f"predict training data... #iter = {train_iter} ({len(ds_preprocessed['train'])} samples / {train_batch_size} batches)")
        train_pred = trainer.predict(ds_preprocessed["train"])
        # repair
        print(f"predict repair data... #iter = {repair_iter} ({len(ds_preprocessed['repair'])} samples / {eval_batch_size} batches)")
        repair_pred = trainer.predict(ds_preprocessed["repair"])
        # test
        print(f"predict test data... #iter = {test_iter} ({len(ds_preprocessed['test'])} samples / {eval_batch_size} batches)")
        test_pred = trainer.predict(ds_preprocessed["test"])
        # Display logs temporarily
        for key, pred in zip(["train", "repair", "test"], [train_pred, repair_pred, test_pred]):
            logger.info(f'metrics for {key} set:\n {pred.metrics}')

        # Save PredictionOutput object containing prediction results with pickle
        with open(os.path.join(pred_out_dir, "train_pred.pkl"), "wb") as f:
            pickle.dump(train_pred, f)
        with open(os.path.join(pred_out_dir, "repair_pred.pkl"), "wb") as f:
            pickle.dump(repair_pred, f)
        with open(os.path.join(pred_out_dir, "test_pred.pkl"), "wb") as f:
            pickle.dump(test_pred, f)
        # Also save proba (prediction probability for each sample) grouped by correct label
        train_labels = np.array(ds["train"][label_col]) # Correct label for each sample
        repair_labels = np.array(ds["repair"][label_col])
        test_labels = np.array(ds["test"][label_col])
        # Convert PredictionOutput object -> prediction probability
        train_pred_proba = pred_to_proba(train_pred)
        repair_pred_proba = pred_to_proba(repair_pred)
        test_pred_proba = pred_to_proba(test_pred)
        # Save as different files by label (train)
        for c in range(len(labels)):
            tgt_proba = train_pred_proba[train_labels == c]
            # Save train_pred_proba
            np.save(os.path.join(pretrained_dir, "pred_results", f"train_proba_{c}.npy"), tgt_proba)
            print(f"train_proba_{c}.npy ({tgt_proba.shape}) saved")
        # Save as different files by label (repair)
        for c in range(len(labels)):
            tgt_proba = repair_pred_proba[repair_labels == c]
            # Save repair_pred_proba
            np.save(os.path.join(pretrained_dir, "pred_results", f"repair_proba_{c}.npy"), tgt_proba)
            print(f"repair_proba_{c}.npy ({tgt_proba.shape}) saved")
        # Save as different files by label (test)
        for c in range(len(labels)):
            tgt_proba = test_pred_proba[test_labels == c]
            # Save test_pred_proba
            np.save(os.path.join(pretrained_dir, "pred_results", f"test_proba_{c}.npy"), tgt_proba)
            print(f"test_proba_{c}.npy ({tgt_proba.shape}) saved")

    # Inference on C10C dataset
    # ====================================================================
    if ds_name == "c10c" or ds_name == "c100c":
        # Loop for 20 types of corruptions
        for key in ds_preprocessed.keys():
            eval_iter = math.ceil(len(ds_preprocessed[key]) / eval_batch_size)
            # Execute inference
            print(f"predict {ds_name}:{key} data... #iter = {eval_iter} ({len(ds_preprocessed[key])} samples / {eval_batch_size} batches)")
            key_pred = trainer.predict(ds_preprocessed[key])
            # Save PredictionOutput object containing prediction results with pickle
            with open(os.path.join(pred_out_dir, f"{ds_name}_{key}_pred.pkl"), "wb") as f:
                pickle.dump(key_pred, f)
