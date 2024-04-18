from datasets import load_metric
from transformers import ViTImageProcessor

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
met_acc = load_metric("accuracy")
met_f1 = load_metric("f1")

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