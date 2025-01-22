import os, sys, math
import numpy as np
import argparse
import torch
import pickle
from datasets import load_from_disk
from transformers import DefaultDataCollator, ViTForImageClassification, Trainer
from utils.helper import get_device, get_corruption_types
from utils.vit_util import processor, transforms, compute_metrics, transforms_c100
from utils.constant import ViTExperiment

if __name__ == "__main__":
    # データセットをargparseで受け取る
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str)
    parser.add_argument("--ct", type=str, help="corruption type for fine-tuned dataset. when set to None, original model is used", default=None)
    parser.add_argument('--severity', type=int, help="severity of corruption (integer from 0 to 4). when set to -1, treat all as one dataset.", default=4)
    args = parser.parse_args()
    ds_name = args.ds
    ori_ct = args.ct
    severity = args.severity
    print(f"ds_name: {ds_name}, ori_ct: {ori_ct}, severity: {severity}")

    # datasetごとに違う変数のセット
    if ds_name == "c10c":
        tf_func = transforms
        label_col = "label"
    elif ds_name == "c100c":
        tf_func = transforms_c100
        label_col = "fine_label"
    else:
        NotImplementedError
    
    # デバイス (cuda, or cpu) の取得
    device = get_device()
    ct_list = get_corruption_types()
    dataset_dir = ViTExperiment.DATASET_DIR
    data_collator = DefaultDataCollator()
    pretrained_dir = getattr(ViTExperiment, ds_name.rstrip('c')).OUTPUT_DIR 

    # 対象のcorruptionでfine-tuneしたモデルをロード
    if ori_ct is not None:
        adapt_out_dir = os.path.join(pretrained_dir, f"{ori_ct}_severity{severity}")
        model = ViTForImageClassification.from_pretrained(adapt_out_dir).to(device)
        training_args = torch.load(os.path.join(adapt_out_dir, "training_args.bin"))
    else:
        model = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
        training_args = torch.load(os.path.join(pretrained_dir, "training_args.bin"))
    # 予測結果格納ディレクトリ
    pred_out_dir = os.path.join(adapt_out_dir, "pred_results", "PredictionOutput") if ori_ct is not None \
            else os.path.join(pretrained_dir, "pred_results_divided_corr", "PredictionOutput")
    os.makedirs(pred_out_dir, exist_ok=True)

    # 各ctのtest setに対する予測を行う
    for i, ct in enumerate(ct_list):
        print(f"{'='*60}\ncorrp_type: {ct} ({i+1}/{len(ct_list)})\n{'='*60}")
        # 対象のcorruption, severityのdatasetをロード
        ds = load_from_disk(os.path.join(dataset_dir, f"{ds_name}_severity{severity}", ct))
        # 読み込まれた時にリアルタイムで前処理を適用するようにする
        ds_preprocessed = ds.with_transform(tf_func)
        # datasetをtrain, testに分ける
        ds_split = ds_preprocessed.train_test_split(test_size=0.4, shuffle=True, seed=777) # XXX: !SEEDは絶対固定!
        ds_train, ds_test = ds_split["train"], ds_split["test"]
        # trainerオブジェクトを作成
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            train_dataset=ds_train,
            eval_dataset=ds_test,
            tokenizer=processor,
        )
        # データセットのサイズとバッチサイズからイテレーション回数を計算
        eval_batch_size = training_args.per_device_eval_batch_size
        eval_iter = math.ceil(len(ds_test) / eval_batch_size)
        # テストデータに対する推論実行
        print(f"predict {ds_name}:{ct} data... #iter = {eval_iter} ({len(ds_test)} samples / {eval_batch_size} batches)")
        pred = trainer.predict(ds_test)
        # 予測結果を格納するPredictionOutputオブジェクトをpickleで保存
        with open(os.path.join(pred_out_dir, f"{ds_name}_{ct}_pred.pkl"), "wb") as f:
            pickle.dump(pred, f)
        print(f"saved to {os.path.join(pred_out_dir, f'{ds_name}_{ct}_pred.pkl')}")
    # オリジナルのデータセットも予測する
    ori_ds = load_from_disk(os.path.join(dataset_dir, ds_name.rstrip('c')))
    ori_ds_preprocessed = ori_ds.with_transform(tf_func)
    trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            train_dataset=ori_ds_preprocessed["train"],
            eval_dataset=ori_ds_preprocessed["test"],
            tokenizer=processor,
    )
    test_pred = trainer.predict(ori_ds_preprocessed["test"])
    with open(os.path.join(pred_out_dir, "ori_test_pred.pkl"), "wb") as f:
        pickle.dump(test_pred, f)
    print(f"saved to {os.path.join(pred_out_dir, 'ori_test_pred.pkl')}")