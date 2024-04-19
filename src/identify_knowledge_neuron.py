import os, sys, time
from tqdm import tqdm
import numpy as np
import torch
from datasets import load_from_disk
from transformers import ViTForImageClassification
from utils.helper import get_device
from utils.vit_util import transforms
from utils.constant import ViTExperiment

def scaled_input(emb, num_points):
    """
    intermediate statesの重みを1/m (m=0,..,num_points) 倍した行列を作る
    """
    # emb: (1, ffn_size)
    baseline = torch.zeros_like(emb)  # (1, ffn_size)

    step = (emb - baseline) / num_points  # (1, ffn_size)

    res = torch.cat([torch.add(baseline, step * i) for i in range(num_points)], dim=0)  # (num_points, ffn_size)
    return res, step[0]

if __name__ == "__main__":
    # デバイス (cuda, or cpu) の取得
    device = get_device()
    # datasetをロード (初回の読み込みだけやや時間かかる)
    cifar10 = load_from_disk(os.path.join(ViTExperiment.DATASET_DIR, "c10"))
    # 読み込まれた時にリアルタイムで前処理を適用するようにする
    cifar10_preprocessed = cifar10.with_transform(transforms)
    # ラベルを示す文字列のlist
    labels = cifar10_preprocessed["train"].features["label"].names
    # pretrained modelのロード
    pretrained_dir = ViTExperiment.OUTPUT_DIR
    model = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
    model.eval()

    # ひとまず適当にtgt_layerを決めて中間層のニューロンを取得する
    tgt_pos = ViTExperiment.CLS_IDX # tgt_posはCLS_IDXで固定 (intermediate_statesの2次元目の0番目の要素に対応する中間層ニューロン)
    num_points = ViTExperiment.NUM_POINTS # integrated gradientの積分近似の分割数
    
    # record various results
    res_dict = {
        'pred': [],
        'ig_pred': [],
        'ig_gold': [],
        'base': []
    }
    tic = time.perf_counter()
    for tgt_layer in range(model.vit.config.num_hidden_layers):
        print(f"tgt_layer={tgt_layer}")
        ig_gold = None # integrated gradient for gold label
        # loop for the dataset
        for si, entry_dic in tqdm(enumerate(cifar10_preprocessed["train"].iter(batch_size=1)), 
                                total=len(cifar10_preprocessed["train"])): # NOTE: 重みを変える分でバッチ次元使ってるのでデータサンプルにバッチ次元をできない (データのバッチ化ができない)
            if si == 1:
                break
            x, y = entry_dic["pixel_values"].to(device), entry_dic["labels"][0]
            # print("get prediction labels...")
            output = model.forward(x, output_hidden_states=True, output_attentions=True, output_intermediate_states=True)
            
            # get pred label
            logits = output.logits
            y_pred = int(torch.argmax(logits[0, :]))  # scalar
            # print(f"y_true={y}, y_pred={y_pred}")
            res_dict['pred'].append(y_pred)
            
            # get intermediate states
            mid = output.intermediate_states
            tgt_mid = mid[tgt_layer][:, tgt_pos, :]
            
            # get scaled weights
            scaled_weights, weights_step = scaled_input(tgt_mid, num_points)  # (num_points, ffn_size), (ffn_size)
            scaled_weights.requires_grad_(True)
            # print("get integrated gradients...")
            output = model(x, tgt_pos=tgt_pos, tgt_layer=tgt_layer, tmp_score=scaled_weights, tgt_label=y)
            grad = output.gradients
            # this var stores the partial diff. for each scaled weights
            grad = grad.sum(dim=0)  # (ffn_size)
            ig_gold = grad if ig_gold is None else torch.add(ig_gold, grad)  # (ffn_size)
        ig_gold = ig_gold * weights_step  # (ffn_size)
        res_dict['ig_gold'].append(ig_gold.tolist())
        res_dict['base'].append(tgt_mid.squeeze().tolist())
    print(np.array(res_dict['ig_gold']).shape) # (12, 3072) = (レイヤ数, 中間ニューロン数)
    toc = time.perf_counter()
    print(f"***** Costing time: {toc - tic:0.4f} seconds *****")
