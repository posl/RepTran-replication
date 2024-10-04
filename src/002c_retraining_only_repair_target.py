"""
We use the whole repair set in 002a,
but we use the part of repair set used for the repair in 002c.
"""

import os
import torch
import argparse
from datasets import load_from_disk
from transformers import DefaultDataCollator, ViTForImageClassification, Trainer
from utils.helper import get_device, get_corruption_types
from utils.vit_util import processor, transforms, transforms_c100, compute_metrics, identfy_tgt_misclf
from utils.constant import ViTExperiment

def retraining_with_repair_set(ds_name, k, tgt_rank, misclf_type, tgt_split):
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
    # repair に使ったサンプルのindexを取得
    misclf_info_dir = os.path.join(pretrained_dir, "misclf_info")
    _, _, tgt_mis_indices = identfy_tgt_misclf(misclf_info_dir, tgt_split=tgt_split, tgt_rank=tgt_rank, misclf_type=misclf_type)
    # tgt_indicesに対応するサンプルだけ取り出す
    ds_tgt = ds_repair.select(indices=tgt_mis_indices)
    model = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
    model.eval()
    # オリジナルモデルの学習時の設定をロード
    training_args = torch.load(os.path.join(pretrained_dir, "training_args.bin"))
    # オリジナルの学習から設定を少し変える
    adapt_out_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", "retraining_with_only_repair_target")
    os.makedirs(adapt_out_dir, exist_ok=True)
    training_args.output_dir = adapt_out_dir
    # print(training_args.num_epochs)
    if misclf_type == "src_tgt":
        training_args.num_train_epochs = 1 # NOTE: これは対象誤分類のデータの数によるが, src_tgtの場合はデータが少ないので1epochで十分
    else:
        training_args.num_train_epochs = 2
    training_args.logging_strategy = "no"
    # Trainerオブジェクトの作成
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
    parser.add_argument('--tgt_rank', type=int, help="the rank of the target misclassification type", default=1)
    parser.add_argument('--misclf_type', type=str, help="the type of misclassification (src_tgt or tgt or all)", default="src_tgt")
    args = parser.parse_args()
    ds_name = args.ds
    k = args.k
    tgt_rank = args.tgt_rank
    misclf_type = args.misclf_type
    print(f"ds_name: {ds_name}, fold_id: {k}")
    retraining_with_repair_set(ds_name, k, tgt_rank, misclf_type, tgt_split="repair")