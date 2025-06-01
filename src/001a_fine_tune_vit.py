import os
import argparse
from datasets import load_from_disk
from transformers import DefaultDataCollator, ViTForImageClassification, TrainingArguments, Trainer
from utils.helper import get_device
from utils.vit_util import processor, transforms, transforms_c100, compute_metrics, maybe_initialize_repair_weights_
from utils.constant import ViTExperiment

if __name__ == "__main__":
    # データセットをargparseで受け取る
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str)
    parser.add_argument('k', type=int, help="the fold id (0 to K-1)")
    args = parser.parse_args()
    ds_name = args.ds
    k = args.k
    print(f"ds_name: {ds_name}, fold_id: {k}")
    # デバイス (cuda, or cpu) の取得
    device = get_device()
    dataset_dir = ViTExperiment.DATASET_DIR
    # datasetをロード (初回の読み込みだけやや時間かかる)
    exp_obj = getattr(ViTExperiment, ds_name.replace("-", "_"))
    if ds_name == "tiny-imagenet":
        ds = load_from_disk(os.path.join(dataset_dir, "tiny-imagenet-200"))
        output_dir = exp_obj.OUTPUT_DIR
        eval_div = "repair"
    else:
        ds = load_from_disk(os.path.join(dataset_dir, f"{ds_name}_fold{k}"))
        output_dir = exp_obj.OUTPUT_DIR.format(k=k)
        eval_div = "test"

    # datasetごとに違う変数のセット
    if ds_name == "c10" or ds_name == "tiny-imagenet":
        tf_func = transforms
        label_col = "label"
    elif ds_name == "c100":
        tf_func = transforms_c100
        label_col = "fine_label"
    else:
        NotImplementedError
    # 読み込まれた時にリアルタイムで前処理を適用するようにする
    ds_preprocessed = ds.with_transform(tf_func)
    # バッチごとの処理のためのdata_collator
    data_collator = DefaultDataCollator()
    # ラベルを示す文字列のlist
    labels = ds_preprocessed["train"].features[label_col].names
    
    # pretrained modelのロード
    model, loading_info = ViTForImageClassification.from_pretrained(
        ViTExperiment.ViT_PATH,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)},
        output_loading_info=True
    )
    model.to(device).eval()
    model = maybe_initialize_repair_weights_(model, loading_info["missing_keys"])

    # 学習の設定
    batch_size = ViTExperiment.BATCH_SIZE
    logging_steps = len(ds_preprocessed["train"]) // batch_size
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=exp_obj.NUM_EPOCHS,
        learning_rate=2e-4,
        weight_decay=0.01,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        remove_unused_columns=False, # img列がないとエラーになるので必要
        evaluation_strategy="epoch", # エポックの終わりごとにeval_datasetで評価
        logging_strategy="epoch",
        save_strategy="epoch",
        push_to_hub=False,
        disable_tqdm=False,
        log_level="error",
        report_to="tensorboard",
        load_best_model_at_end=True,
    )

    # 学習の実行
    # NOTE: 表示されるプログレスバーの分母の数字は，num_epoch*num_sample/batch_size
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        train_dataset=ds_preprocessed["train"],
        eval_dataset=ds_preprocessed[eval_div],
        tokenizer=processor,
    )
    train_results = trainer.train()

    # 保存
    trainer.save_model() # from_pretrained()から読み込めるようになる
    trainer.save_state() # save_model()よりも多くの情報を保存する
