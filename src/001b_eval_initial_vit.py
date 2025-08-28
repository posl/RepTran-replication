import os, sys, math, time
import numpy as np
import argparse
import torch
import pickle
from datasets import load_from_disk
from transformers import DefaultDataCollator, ViTForImageClassification, Trainer
from utils.helper import get_device
from utils.vit_util import processor, transforms, compute_metrics, transforms_c100, pred_to_proba, maybe_initialize_repair_weights_
from utils.constant import ViTExperiment
from utils.log import set_exp_logging
from logging import getLogger

logger = getLogger("base_logger")

def main(ds_name, k):
    print(f"ds_name: {ds_name}, fold_id: {k}")
    
    # Get device (cuda or cpu)
    device = get_device()
    # Load dataset (takes some time only on first load)
    dataset_dir = ViTExperiment.DATASET_DIR
    # data_collator for batch processing
    data_collator = DefaultDataCollator()
    
    exp_obj = getattr(ViTExperiment, ds_name.replace("-", "_"))
    ds = load_from_disk(os.path.join(dataset_dir, f"{ds_name}_fold{k}"))
    pretrained_dir = exp_obj.OUTPUT_DIR.format(k=k)
    eval_div = "test"
    ds_preprocessed = ds.with_transform(transforms)
    
    # Set different variables for each dataset
    if ds_name == "c10" or ds_name == "c10c" or ds_name == "tiny-imagenet":
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
    this_file_name = os.path.basename(__file__).split(".")[0]
    logger = set_exp_logging(exp_dir=pretrained_dir, exp_name=this_file_name)
    logger.info(f"ds_name: {ds_name}, fold_id: {k}")
    model, loading_info = ViTForImageClassification.from_pretrained(pretrained_dir, output_loading_info=True)
    model = model.to(device) #in-place so assignment is not necessary
    # If intermediate.repair.weight is not in loaded state dict, must initialize here
    maybe_initialize_repair_weights_(model, loading_info["missing_keys"])
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
        eval_dataset=ds_preprocessed[eval_div],
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

    # Inference on C10 or C100 or tiny-imagenet datasets
    # ====================================================================
    if ds_name == "c10" or ds_name == "c100" or ds_name == "tiny-imagenet":
        # Calculate number of iterations from dataset size and batch size
        train_iter = math.ceil(len(ds_preprocessed["train"]) / train_batch_size)
        repair_iter = math.ceil(len(ds_preprocessed["repair"]) / eval_batch_size)
        test_iter = math.ceil(len(ds_preprocessed["test"]) / eval_batch_size)
        # Execute inference on train, repair, test data
        # train
        logger.info(f"predict training data... #iter = {train_iter} ({len(ds_preprocessed['train'])} samples / {train_batch_size} batches)")
        st = time.perf_counter()
        train_pred = trainer.predict(ds_preprocessed["train"])
        et = time.perf_counter()
        t_train = et - st
        logger.info(f"elapsed time: {t_train} sec")
        # repair
        logger.info(f"predict repair data... #iter = {repair_iter} ({len(ds_preprocessed['repair'])} samples / {eval_batch_size} batches)")
        st = time.perf_counter()
        repair_pred = trainer.predict(ds_preprocessed["repair"])
        et = time.perf_counter()
        t_repair = et - st
        logger.info(f"elapsed time: {t_repair} sec")
        # test
        logger.info(f"predict test data... #iter = {test_iter} ({len(ds_preprocessed['test'])} samples / {eval_batch_size} batches)")
        st = time.perf_counter()
        test_pred = trainer.predict(ds_preprocessed["test"])
        et = time.perf_counter()
        t_test = et - st
        logger.info(f"elapsed time: {t_test} sec")
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
            logger.info(f"train_proba_{c}.npy ({tgt_proba.shape}) saved")
        # Save as different files by label (repair)
        for c in range(len(labels)):
            tgt_proba = repair_pred_proba[repair_labels == c]
            # Save repair_pred_proba
            np.save(os.path.join(pretrained_dir, "pred_results", f"repair_proba_{c}.npy"), tgt_proba)
            logger.info(f"repair_proba_{c}.npy ({tgt_proba.shape}) saved")
        # Save as different files by label (test)
        for c in range(len(labels)):
            tgt_proba = test_pred_proba[test_labels == c]
            # Save test_pred_proba
            np.save(os.path.join(pretrained_dir, "pred_results", f"test_proba_{c}.npy"), tgt_proba)
            logger.info(f"test_proba_{c}.npy ({tgt_proba.shape}) saved")

        logger.info(f"all pred_results saved, total elapsed time: {t_train+t_repair+t_test} sec")
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

if __name__ == "__main__":
    # Accept dataset via argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str)
    parser.add_argument('k_list', type=int, nargs="*", default=[0, 1, 2, 3, 4], help="the fold id(s) to run (default: 0 1 2 3 4)")
    args = parser.parse_args()
    ds_name = args.ds
    k_list = args.k_list
    for k in k_list:
        main(ds_name, k)