import os
import torch
import argparse
from datasets import load_from_disk
from transformers import DefaultDataCollator, ViTForImageClassification, Trainer
from utils.helper import get_device, get_corruption_types
from utils.vit_util import processor, transforms, transforms_c100, compute_metrics
from utils.constant import ViTExperiment

def retraining_with_repair_set(ds_name, k):
    # datasetごとに違う変数のセット
    if ds_name == "c10":
        tf_func = transforms
        label_col = "label"
    elif ds_name == "c100":
        tf_func = transforms_c100
        label_col = "fine_label"
    else:
        NotImplementedError

    # デバイス (cuda, or cpu) の取得
    device = get_device()
    dataset_dir = ViTExperiment.DATASET_DIR
    data_collator = DefaultDataCollator()
    ds_dirname = f"{ds_name}_fold{k}"

    # 対象のcorruption, severityのdatasetをロード
    ds = load_from_disk(os.path.join(dataset_dir, ds_dirname))
    # 読み込まれた時にリアルタイムで前処理を適用するようにする
    ds_preprocessed = ds.with_transform(tf_func)
    ds_train, ds_repair, ds_test = ds_preprocessed["train"], ds_preprocessed["repair"], ds_preprocessed["test"]

    # pretrained modelのロード
    pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR.format(k=k)
    model = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
    # オリジナルモデルの学習時の設定をロード
    training_args = torch.load(os.path.join(pretrained_dir, "training_args.bin"))
    # オリジナルの学習から設定を少し変える
    adapt_out_dir = os.path.join(pretrained_dir, "retraining_with_repair_set")
    training_args.output_dir = adapt_out_dir
    training_args.num_epochs = 2
    training_args.logging_strategy = "no"
    # Trainerオブジェクトの作成
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        train_dataset=ds_repair, #NOTE: train setでの訓練は終わっているのでrepair setをつかった追加の訓練
        eval_dataset=ds_test,
        tokenizer=processor,
    )
    # 学習の実行
    train_results = trainer.train()
    # 保存
    trainer.save_model() # from_pretrained()から読み込めるようになる
    trainer.save_state() # save_model()よりも多くの情報を保存する

if __name__ == "__main__":
    # データセットをargparseで受け取る
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str, help="dataset name")
    parser.add_argument('k', type=int, help="the fold id (0 to K-1)")
    args = parser.parse_args()
    ds_name = args.ds
    k = args.k
    print(f"ds_name: {ds_name}, fold_id: {k}")
    retraining_with_repair_set(ds_name, k)