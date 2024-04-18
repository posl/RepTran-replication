import os, sys
import numpy as np
import torch
from transformers import AutoModelForImageClassification, Trainer
from utils.helper import get_device

if __name__ == "__main__":
    device = get_device()
    pretrained_dir = "/src/src/out_vit_c10"
    # pretrained modelのロード
    loaded_model = AutoModelForImageClassification.from_pretrained(pretrained_dir).to(device)
    loaded_model.eval()
    
    # パラメータ数のカウント
    # ==============================================================
    num_params = 0
    skipped_layer = ["layernorm"] # パラメータ数をカウントしないレイヤのリスト
    # レイヤの種類ごとのパラメータ数集計用
    layer_types = { # このdictは，レイヤ名とレイヤ種類の対応を示す．valのlistの最初のlistはホワイトリストで，2番目のlistはブラックリスト
        "attention": [["attention"], []],
        "intermediate": [["intermediate", "output"], ["attention"]],
        "embeddings": [["embeddings"], []],
    }
    layer_params = {layer: 0 for layer in layer_types.keys()}
    for name, param in loaded_model.named_parameters():
        # print(name, param.shape)
        # nameにskipped_layerのいずれかの文字列が含まれていたらスキップ
        if any(layer in name for layer in skipped_layer):
            continue
        # 総パラメータ数のカウント
        num_params += np.prod(param.shape)
        for layer, (white_lst, black_lst) in layer_types.items():
            # nameにwhite_lstのいずれかの文字列が含まれており，かつblack_lstのいずれの文字列も含まれていなかったら，layer_params[layer]に加算
            if any(white in name for white in white_lst) and not any(black in name for black in black_lst):
                layer_params[layer] += np.prod(param.shape)
    # 総パラメータ数
    print(f"num_params = {num_params} (={num_params/1e6}M)\n{'='*50}")
    # レイヤごとのパラメータ数
    for layer, params in layer_params.items():
        print(f"params of {layer} = {params} (={params/1e6}M, {params/num_params*100:.2f}%)")
    # ==============================================================