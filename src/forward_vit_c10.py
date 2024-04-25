import os, sys, math
import numpy as np
import argparse
import torch
import pickle
from datasets import load_from_disk
from transformers import DefaultDataCollator, ViTForImageClassification, Trainer
from utils.helper import get_device
from utils.vit_util import processor, transforms, compute_metrics
from utils.constant import ViTExperiment

def pred_to_proba(pred):
    proba = torch.nn.functional.softmax(torch.tensor(pred.predictions[1]), dim=-1)
    return proba.cpu().numpy()


if __name__ == "__main__":
    # c10についての推論を行うかどうか，c10cについての推論を行うかどうかをargparseで指定
    parser = argparse.ArgumentParser()
    parser.add_argument("--c10", action="store_true", help="predict CIFAR-10 dataset")
    parser.add_argument("--c10c", action="store_true", help="predict CIFAR-10-C dataset")
    args = parser.parse_args()
    # どちらも指定されていない場合はエラー
    if not args.c10 and not args.c10c:
        print("Please specify either --c10 or --c10c.")
        sys.exit(1)
    
    # デバイス (cuda, or cpu) の取得
    device = get_device()
    # datasetをロード (初回の読み込みだけやや時間かかる)
    dataset_dir = ViTExperiment.DATASET_DIR
    cifar10 = load_from_disk(os.path.join(dataset_dir, "c10"))
    # 読み込まれた時にリアルタイムで前処理を適用するようにする
    cifar10_preprocessed = cifar10.with_transform(transforms)
    # バッチごとの処理のためのdata_collator
    data_collator = DefaultDataCollator()
    # ラベルを示す文字列のlist
    labels = cifar10_preprocessed["train"].features["label"].names
    # pretrained modelのロード
    pretrained_dir = ViTExperiment.OUTPUT_DIR
    model = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
    model.eval()
    # 学習時の設定をロード
    training_args = torch.load(os.path.join(pretrained_dir, "training_args.bin"))
    # Trainerオブジェクトの作成
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        train_dataset=cifar10_preprocessed["train"],
        eval_dataset=cifar10_preprocessed["test"],
        tokenizer=processor,
    )
    # データセットのサイズとバッチサイズからイテレーション回数を計算
    training_args_dict = training_args.to_dict()
    train_batch_size = training_args_dict["per_device_train_batch_size"]
    eval_batch_size = training_args_dict["per_device_eval_batch_size"]
    train_iter = math.ceil(len(cifar10_preprocessed["train"]) / train_batch_size)
    eval_iter = math.ceil(len(cifar10_preprocessed["test"]) / eval_batch_size)
    # 予測結果格納ディレクトリ
    pred_out_dir = os.path.join(pretrained_dir, "pred_results", "PredictionOutput")
    # pred_out_dirがなければ作成
    if not os.path.exists(pred_out_dir):
        os.makedirs(pred_out_dir)

    # オリジナルのC10データセットに対する推論
    # ====================================================================
    if args.c10:
        # 訓練・テストデータに対する推論の実行
        print(f"predict training data... #iter = {train_iter} ({len(cifar10_preprocessed['train'])} samples / {train_batch_size} batches)")
        train_pred = trainer.predict(cifar10_preprocessed["train"])
        print(f"predict evaluation data... #iter = {eval_iter} ({len(cifar10_preprocessed['test'])} samples / {eval_batch_size} batches)")
        test_pred = trainer.predict(cifar10_preprocessed["test"])
        # 予測結果を格納するPredictioOutputオブジェクトをpickleで保存
        with open(os.path.join(pred_out_dir, "train_pred.pkl"), "wb") as f:
            pickle.dump(train_pred, f)
        with open(os.path.join(pred_out_dir, "test_pred.pkl"), "wb") as f:
            pickle.dump(test_pred, f)
        # proba (各サンプルに対する予測確率) を正解ラベルごとにまとめたものも保存する
        train_labels = np.array(cifar10["train"]["label"])
        test_labels = np.array(cifar10["test"]["label"])
        # PredictionOutputオブジェクト -> 予測確率に変換
        train_pred_proba = pred_to_proba(train_pred)
        test_pred_proba = pred_to_proba(test_pred)
        # ラベルごとに違うファイルとして保存 (train)
        for c in range(len(labels)):
            tgt_proba = train_pred_proba[train_labels == c]
            # train_pred_probaを保存
            np.save(os.path.join(pretrained_dir, "pred_results", f"train_proba_{c}.npy"), tgt_proba)
            print(f"train_proba_{c}.npy ({tgt_proba.shape}) saved")
        # ラベルごとに違うファイルとして保存 (test)
        for c in range(len(labels)):
            tgt_proba = test_pred_proba[test_labels == c]
            # test_pred_probaを保存
            np.save(os.path.join(pretrained_dir, "pred_results", f"test_proba_{c}.npy"), tgt_proba)
            print(f"test_proba_{c}.npy ({tgt_proba.shape}) saved")

    # C10Cデータセットに対する推論
    # ====================================================================
    if args.c10c:
        # C10Cデータセットのロード
        c10c = load_from_disk(os.path.join(dataset_dir, "c10c"))
        # 20種類のcorruptionsに対するループ
        for key in c10c.keys():
            cifar10_preprocessed = c10c[key].with_transform(transforms)
            # 推論の実行
            print(f"predict c10c:{key} data... #iter = {eval_iter} ({len(cifar10_preprocessed)} samples / {eval_batch_size} batches)")
            key_pred = trainer.predict(cifar10_preprocessed)
            # 予測結果を格納するPredictioOutputオブジェクトをpickleで保存
            with open(os.path.join(pred_out_dir, f"c10c_{key}_pred.pkl"), "wb") as f:
                pickle.dump(key_pred, f)