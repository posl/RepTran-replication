import os
import json
import numpy as np
import torch
import pickle
from datasets import load_from_disk
from transformers import DefaultDataCollator, ViTForImageClassification, Trainer
from utils.helper import get_device
from utils.vit_util import processor, transforms_c100, compute_metrics
from utils.constant import ViTExperiment, Experiment
from tqdm import tqdm
from collections import defaultdict

if __name__ == "__main__":
    model_dir = ViTExperiment.c100.OUTPUT_DIR.format(k=0)

    # Setup device and model
    device = get_device()
    model = ViTForImageClassification.from_pretrained(model_dir).to(device)
    model.eval()
    
    # Load training args
    training_args = torch.load(os.path.join(model_dir, "training_args.bin"))
    
    # Setup Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DefaultDataCollator(),
        compute_metrics=compute_metrics,
        tokenizer=processor,
    )
    
    # Load CIFAR-100-C dataset
    dataset_dir = Experiment.DATASET_DIR
    ds = load_from_disk(os.path.join(dataset_dir, "c100c"))
    ds_preprocessed = ds.with_transform(transforms_c100)

    # 結果格納用
    accuracy_dict = {}
    metrics_dict = {}
    error_index_dict = {}

    # 追加のノイズはスキップする
    additional_noise = ViTExperiment.c100.ADDITIONAL_NOISE_TYPES

    # tqdmで進捗表示しながら処理
    for key in tqdm(ds_preprocessed.keys(), desc="Evaluating corruptions"):
        if key in additional_noise:
            print(f"Skipping additional noise type: {key}")
            continue

        pred = trainer.predict(ds_preprocessed[key])
        pred_out_save_path = os.path.join(model_dir, "pred_results", "PredictionOutput", f"{key}_pred.pkl")
        os.makedirs(os.path.dirname(pred_out_save_path), exist_ok=True)
        # 予測結果を格納するPredictioOutputオブジェクトをpickleで保存
        with open(pred_out_save_path, "wb") as f:
            pickle.dump(pred, f)
        print(f"Saved prediction output to: {pred_out_save_path}")

        # Accuracy取得
        try:
            acc = pred.metrics["test_accuracy"]["accuracy"]
            accuracy_dict[key] = acc
            metrics_dict[key] = pred.metrics
        except KeyError:
            # この場合はエラー終了したい
            raise KeyError(f"No accuracy found in prediction for {key}")

        # 誤分類のインデックスを記録
        preds = np.argmax(pred.predictions, axis=1)
        labels = pred.label_ids
        wrong_indices = np.where(preds != labels)[0].tolist()
        error_index_dict[key] = wrong_indices

    # JSONで保存
    output_acc_path = os.path.join(model_dir, "corruption_accuracy.json")
    output_met_path = os.path.join(model_dir, "corruption_metrics.json")
    output_idx_path = os.path.join(model_dir, "corruption_error_indices.json")

    with open(output_acc_path, "w") as f:
        json.dump(accuracy_dict, f, indent=2)
        
    with open(output_met_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)

    with open(output_idx_path, "w") as f:
        json.dump(error_index_dict, f, indent=2)

    print(f"\nSaved accuracy to: {output_acc_path}")
    print(f"Saved metrics to: {output_met_path}")
    print(f"Saved error indices to: {output_idx_path}")
    
    # ===== サマリー出力 =====
    print("\n===== Summary =====")
    for key in sorted(accuracy_dict.keys()):
        acc = accuracy_dict[key]
        wrong_indices = error_index_dict.get(key, [])
        num_errors = len(wrong_indices)
        total = len(ds_preprocessed[key])
        num_correct = total - num_errors

        # 整合性チェック
        assert num_correct + num_errors == total, f"Mismatch in counts for {key}"

        print(f"{key:<20}: accuracy = {acc:.3f} ({num_correct}/{total}), errors = {num_errors}")
    print("===================\n")
    
    # 結果格納用
    severity_error_counts = defaultdict(lambda: [0]*5)  # {noise_type: [s0, s1, s2, s3, s4]}

    # 範囲設定
    severity_ranges = [(i*10000, (i+1)*10000) for i in range(5)]

    for noise_type, indices in error_index_dict.items():
        for idx in indices:
            for severity, (start, end) in enumerate(severity_ranges):
                if start <= idx < end:
                    severity_error_counts[noise_type][severity] += 1
                    break
    
    # 出力
    print("=== Error Counts by Severity ===")
    for noise_type in sorted(severity_error_counts):
        counts = severity_error_counts[noise_type]
        max_sev = max(range(5), key=lambda i: counts[i])
        counts_str = ", ".join(f"s{lvl}={cnt}" for lvl, cnt in enumerate(counts))
        print(f"{noise_type:<20}: {counts_str} -> max: s{max_sev} ({counts[max_sev]} errors)")
    print("===================\n")
