import os
import torch
import argparse
from datasets import load_from_disk
from transformers import DefaultDataCollator, ViTForImageClassification, Trainer
from utils.helper import get_device, get_corruption_types
from utils.vit_util import processor, transforms, transforms_c100, compute_metrics
from utils.constant import ViTExperiment

def fine_tune_corruption(ds_name, ct, severity):
    # Set different variables for each dataset
    if ds_name == "c10c":
        tf_func = transforms
        label_col = "label"
    elif ds_name == "c100c":
        tf_func = transforms_c100
        label_col = "fine_label"
    else:
        NotImplementedError

    # Get device (cuda or cpu)
    device = get_device()
    dataset_dir = ViTExperiment.DATASET_DIR
    data_collator = DefaultDataCollator()

    # 対象のcorruption, severityのdatasetをロード
    ds = load_from_disk(os.path.join(dataset_dir, f"{ds_name}_severity{severity}", ct))
    # Apply preprocessing in real-time when loaded
    ds_preprocessed = ds.with_transform(tf_func)
    # datasetをtrain, testに分ける
    ds_split = ds_preprocessed.train_test_split(test_size=0.4, shuffle=True, seed=777) # XXX: !SEEDは絶対固定!
    ds_train, ds_test = ds_split["train"], ds_split["test"]

    # Load pretrained model
    pretrained_dir = getattr(ViTExperiment, ds_name.rstrip('c')).OUTPUT_DIR
    model = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
    # オリジナルモデルのLoad training configuration
    training_args = torch.load(os.path.join(pretrained_dir, "training_args.bin"))
    # オリジナルの学習から設定を少し変える
    adapt_out_dir = os.path.join(pretrained_dir, f"{ct}_severity{severity}")
    training_args.output_dir = adapt_out_dir
    training_args.num_epochs = 2
    training_args.logging_strategy = "no"
    # Create Trainer object
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        train_dataset=ds_train,
        eval_dataset=ds_test,
        tokenizer=processor,
    )
    # Execute training
    train_results = trainer.train()
    # Save
    trainer.save_model() # Can be loaded with from_pretrained()
    trainer.save_state() # save_model()よりも多くの情報をSaveする

if __name__ == "__main__":
    # Accept dataset via argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str, help="dataset name (c10c or c100c)")
    parser.add_argument("--ct", type=str, help="corruption type for dataset", default=None)
    parser.add_argument('--severity', type=int, help="severity of corruption (integer from 0 to 4). when set to -1, treat all as one dataset.", default=4)
    args = parser.parse_args()
    ds_name = args.ds
    ct = args.ct
    severity = args.severity
    if ct is not None:
        print(f"{'='*60}\nds_name: {ds_name}, corrp_type: {ct}\n{'='*60}")
        # training for corruption dataset
        fine_tune_corruption(ds_name, ct, severity)
    else:
        print(f"{'='*60}\nds_name: {ds_name}, corrp_type: all\n{'='*60}")
        # get all corrup. type
        ct_list = get_corruption_types()
        for i, ct in enumerate(ct_list):
            print(f"{'='*60}\ncorrp_type: {ct} ({i+1}/{len(ct_list)})\n{'='*60}")
            # training for corruption dataset
            fine_tune_corruption(ds_name, ct, severity)