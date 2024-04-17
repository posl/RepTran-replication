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
    for name, param in loaded_model.named_parameters():
        # nameにskipped_layerのいずれかの文字列が含まれていたらスキップ
        if any(layer in name for layer in skipped_layer):
            continue
        num_params += np.prod(param.shape)
    print(f"num_params = {num_params} (={num_params/1e6}M)")
    # ==============================================================