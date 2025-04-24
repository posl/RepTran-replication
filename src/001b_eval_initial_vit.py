import os, sys, math, time
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

def main(ds_name, k):
    print(f"ds_name: {ds_name}, fold_id: {k}")
    
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
    pretrained_dir = getattr(ViTExperiment, ds_ori_name).OUTPUT_DIR.format(k=k)
    this_file_name = os.path.basename(__file__).split(".")[0]
    logger = set_exp_logging(exp_dir=pretrained_dir, exp_name=this_file_name)
    logger.info(f"ds_name: {ds_name}, fold_id: {k}")
    model = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
    # intermediate.repairレイヤの恒等初期化を明示的にやり直す (訓練した時はrepairなかったので)
    with torch.no_grad():
        for i in range(len(model.vit.encoder.layer)):
            model.vit.encoder.layer[i].intermediate.repair.weight.copy_(torch.eye(3072))
            model.vit.encoder.layer[i].intermediate.repair.weight.requires_grad = False
    model.eval()
    # 学習時の設定をロード
    training_args = torch.load(os.path.join(pretrained_dir, "training_args.bin"))
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
    pred_out_dir = os.path.join(pretrained_dir, "pred_results", "PredictionOutput")
    # pred_out_dirがなければ作成
    if not os.path.exists(pred_out_dir):
        os.makedirs(pred_out_dir)

    # C10 or C100データセットに対する推論
    # ====================================================================
    if ds_name == "c10" or ds_name == "c100":
        # データセットのサイズとバッチサイズからイテレーション回数を計算
        train_iter = math.ceil(len(ds_preprocessed["train"]) / train_batch_size)
        repair_iter = math.ceil(len(ds_preprocessed["repair"]) / eval_batch_size)
        test_iter = math.ceil(len(ds_preprocessed["test"]) / eval_batch_size)
        # train, repair, testデータに対する推論の実行
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
        # いったんログ表示
        for key, pred in zip(["train", "repair", "test"], [train_pred, repair_pred, test_pred]):
            logger.info(f'metrics for {key} set:\n {pred.metrics}')

        # 予測結果を格納するPredictioOutputオブジェクトをpickleで保存
        with open(os.path.join(pred_out_dir, "train_pred.pkl"), "wb") as f:
            pickle.dump(train_pred, f)
        with open(os.path.join(pred_out_dir, "repair_pred.pkl"), "wb") as f:
            pickle.dump(repair_pred, f)
        with open(os.path.join(pred_out_dir, "test_pred.pkl"), "wb") as f:
            pickle.dump(test_pred, f)
        # proba (各サンプルに対する予測確率) を正解ラベルごとにまとめたものも保存する
        train_labels = np.array(ds["train"][label_col]) # サンプルごとの正解ラベル
        repair_labels = np.array(ds["repair"][label_col])
        test_labels = np.array(ds["test"][label_col])
        # PredictionOutputオブジェクト -> 予測確率に変換
        train_pred_proba = pred_to_proba(train_pred)
        repair_pred_proba = pred_to_proba(repair_pred)
        test_pred_proba = pred_to_proba(test_pred)
        # ラベルごとに違うファイルとして保存 (train)
        for c in range(len(labels)):
            tgt_proba = train_pred_proba[train_labels == c]
            # train_pred_probaを保存
            np.save(os.path.join(pretrained_dir, "pred_results", f"train_proba_{c}.npy"), tgt_proba)
            logger.info(f"train_proba_{c}.npy ({tgt_proba.shape}) saved")
        # ラベルごとに違うファイルとして保存 (repair)
        for c in range(len(labels)):
            tgt_proba = repair_pred_proba[repair_labels == c]
            # repair_pred_probaを保存
            np.save(os.path.join(pretrained_dir, "pred_results", f"repair_proba_{c}.npy"), tgt_proba)
            logger.info(f"repair_proba_{c}.npy ({tgt_proba.shape}) saved")
        # ラベルごとに違うファイルとして保存 (test)
        for c in range(len(labels)):
            tgt_proba = test_pred_proba[test_labels == c]
            # test_pred_probaを保存
            np.save(os.path.join(pretrained_dir, "pred_results", f"test_proba_{c}.npy"), tgt_proba)
            logger.info(f"test_proba_{c}.npy ({tgt_proba.shape}) saved")

        logger.info(f"all pred_results saved, total elapsed time: {t_train+t_repair+t_test} sec")
    # C10Cデータセットに対する推論
    # ====================================================================
    if ds_name == "c10c" or ds_name == "c100c":
        # 20種類のcorruptionsに対するループ
        for key in ds_preprocessed.keys():
            eval_iter = math.ceil(len(ds_preprocessed[key]) / eval_batch_size)
            # 推論の実行
            print(f"predict {ds_name}:{key} data... #iter = {eval_iter} ({len(ds_preprocessed[key])} samples / {eval_batch_size} batches)")
            key_pred = trainer.predict(ds_preprocessed[key])
            # 予測結果を格納するPredictionOutputオブジェクトをpickleで保存
            with open(os.path.join(pred_out_dir, f"{ds_name}_{key}_pred.pkl"), "wb") as f:
                pickle.dump(key_pred, f)

if __name__ == "__main__":
    # データセットをargparseで受け取る
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str)
    parser.add_argument('k_list', type=int, nargs="*", default=[0, 1, 2, 3, 4], help="the fold id(s) to run (default: 0 1 2 3 4)")
    args = parser.parse_args()
    ds_name = args.ds
    k_list = args.k_list
    for k in k_list:
        main(ds_name, k)