import os
from datasets import load_from_disk
from transformers import DefaultDataCollator, ViTForImageClassification, TrainingArguments, Trainer
from utils.helper import get_device
from utils.vit_util import processor, transforms, compute_metrics
from utils.constant import ViTExperiment

if __name__ == "__main__":
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
    model = ViTForImageClassification.from_pretrained(
        ViTExperiment.ViT_PATH,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)}
    ).to(device)

    # 学習の設定
    batch_size = ViTExperiment.BATCH_SIZE
    logging_steps = len(cifar10_preprocessed["train"]) // batch_size
    training_args = TrainingArguments(
        output_dir=ViTExperiment.OUTPUT_DIR,
        num_train_epochs=ViTExperiment.NUM_EPOCHS,
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
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        train_dataset=cifar10_preprocessed["train"],
        eval_dataset=cifar10_preprocessed["test"],
        tokenizer=processor,
    )
    train_results = trainer.train()

    # 保存
    trainer.save_model() # from_pretrained()から読み込めるようになる
    trainer.save_state() # save_model()よりも多くの情報を保存する
