import torch
from datasets import load_dataset, load_metric
from transformers import ViTImageProcessor, DefaultDataCollator, ViTForImageClassification, TrainingArguments, Trainer
from utils import get_device

def transforms(batch):
    """
    画像のバッチを前処理する
    
    Parameters
    ------------------
    
    Returns
    ------------------
    
    """
    # 画像のバッチを変換してtorch.tensorにする
    inputs = processor(images=batch["img"], return_tensors="pt")

    # ラベルのフィールドも前処理時に追加
    inputs["labels"] = batch["label"]
    return inputs

def compute_metrics(pred):
    """
    予測結果から評価指標を計算する
    
    Parameters
    ------------------
    
    Returns
    ------------------
    
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = met_acc.compute(predictions=preds, references=labels)
    f1 = met_f1.compute(predictions=preds, references=labels, average="macro")
    return {
        "accuracy": acc,
        "f1": f1
    }

# HACK: processorだけは仕方なくグローバル定義
# 今回用いるViTで使われている前処理をロード
model_name_or_path = 'google/vit-base-patch16-224-in21k'
processor = ViTImageProcessor.from_pretrained(model_name_or_path)

if __name__ == "__main__":
    # デバイス (cuda, or cpu) の取得
    device = get_device()
    # datasetをロード (初回の読み込みだけやや時間かかる)
    cifar10 = load_dataset("cifar10")
    # 読み込まれた時にリアルタイムで前処理を適用するようにする
    cifar10_preprocessed = cifar10.with_transform(transforms)
    # バッチごとの処理のためのdata_collator
    data_collator = DefaultDataCollator()
    # 評価指標のロード
    met_acc = load_metric("accuracy")
    met_f1 = load_metric("f1")
    # ラベルを示す文字列のlist
    labels = cifar10_preprocessed["train"].features["label"].names
    # pretrained modelのロード
    model = ViTForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)}
    ).to(device)

    # 学習の設定
    batch_size = 32
    logging_steps = len(cifar10_preprocessed["train"]) // batch_size
    training_args = TrainingArguments(
        output_dir="./out_vit_c10",
        num_train_epochs=2,
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

    # 学習＆テストデータの予測
    train_pred = trainer.predict(cifar10_preprocessed["train"])
    test_pred = trainer.predict(cifar10_preprocessed["test"])
    print(train_pred.metrics)
    print(test_pred.metrics)

    # 保存
    trainer.save_model() # from_pretrained()から読み込めるようになる
    trainer.save_state() # save_model()よりも多くの情報を保存する
