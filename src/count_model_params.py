import os, sys
import numpy as np
import torch
from transformers import AutoModelForImageClassification, Trainer
from utils import get_device

if __name__ == "__main__":
    device = get_device()
    pretrained_dir = "/src/src/out_vit_c10"
    # pretrained modelのロード
    loaded_model = AutoModelForImageClassification.from_pretrained(pretrained_dir).to(device)
    loaded_model.eval()
    # パラメータ数のカウント
    # ==============================================================
    num_params = 0
    # パラメータ数をカウントしないレイヤのリスト
    skipped_layer = ["layernorm"]
    # レイヤの種類ごとのパラメータ数集計用
    layer_types = {
        "attention": ["attention"],
        "intermediate": ["intermediate", "output"],
        "embeddings": ["embeddings"],
    }
    layer_params = {layer: 0 for layer in layer_types.keys()}
    for name, param in loaded_model.named_parameters():
        # nameにskipped_layerのいずれかの文字列が含まれていたらスキップ
        if any(layer in name for layer in skipped_layer):
            continue
        num_params += np.prod(param.shape)
        for layer, keywords in layer_types.items():
            if any(keyword in name for keyword in keywords):
                layer_params[layer] += np.prod(param.shape)
    # 総パラメータ数
    print(f"num_params = {num_params} (={num_params/1e6}M)\n{'='*50}")
    # レイヤごとのパラメータ数
    for layer, params in layer_params.items():
        print(f"params of {layer} = {params} (={params/1e6}M, {params/num_params*100:.2f}%)")
    # ==============================================================