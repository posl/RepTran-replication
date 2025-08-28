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


if __name__ == "__main__":
    # Accept dataset via argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str)
    args = parser.parse_args()
    ds_name = args.ds
    print(f"ds_name: {ds_name}")
    
    # Get device (cuda or cpu)
    device = get_device()
    # Load dataset (takes some time only on first load)
    dataset_dir = ViTExperiment.DATASET_DIR
    # data_collator for batch processing
    data_collator = DefaultDataCollator()
    ds = load_from_disk(os.path.join(dataset_dir, ds_name))
    # label取得のためoriginal datasetも取得する
    ds_ori_name = ds_name.rstrip("c") if ds_name.endswith("c") else ds_name
    ds_ori = load_from_disk(os.path.join(dataset_dir, ds_ori_name))
    ds_ori_preprocessed = ds_ori.with_transform(transforms)
    
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
    labels = ds_ori_preprocessed["train"].features[label_col].names
    
    # Load pretrained model
    pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR if ds_name == "c10" or ds_name == "c100" else getattr(ViTExperiment, ds_ori_name).OUTPUT_DIR
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
        train_dataset=ds_ori_preprocessed["train"],
        eval_dataset=ds_ori_preprocessed["test"],
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
        eval_iter = math.ceil(len(ds_preprocessed["test"]) / eval_batch_size)
        # 訓練・テストデータに対するExecute inference
        print(f"predict training data... #iter = {train_iter} ({len(ds_preprocessed['train'])} samples / {train_batch_size} batches)")
        train_pred = trainer.predict(ds_preprocessed["train"])
        print(f"predict evaluation data... #iter = {eval_iter} ({len(ds_preprocessed['test'])} samples / {eval_batch_size} batches)")
        test_pred = trainer.predict(ds_preprocessed["test"])
        # Save PredictionOutput object containing prediction results with pickle
        with open(os.path.join(pred_out_dir, "train_pred.pkl"), "wb") as f:
            pickle.dump(train_pred, f)
        with open(os.path.join(pred_out_dir, "test_pred.pkl"), "wb") as f:
            pickle.dump(test_pred, f)
        # Also save proba (prediction probability for each sample) grouped by correct label
        train_labels = np.array(ds["train"][label_col]) # Correct label for each sample
        test_labels = np.array(ds["test"][label_col])
        # Convert PredictionOutput object -> prediction probability
        train_pred_proba = pred_to_proba(train_pred)
        test_pred_proba = pred_to_proba(test_pred)
        # Save as different files by label (train)
        for c in range(len(labels)):
            tgt_proba = train_pred_proba[train_labels == c]
            # Save train_pred_proba
            np.save(os.path.join(pretrained_dir, "pred_results", f"train_proba_{c}.npy"), tgt_proba)
            print(f"train_proba_{c}.npy ({tgt_proba.shape}) saved")
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
