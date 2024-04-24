import os, sys, time
from tqdm import tqdm
import numpy as np
import torch
import argparse
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
    # プログラム引数の受け取り
    parser = argparse.ArgumentParser(description='start_layer_idx selector')
    parser.add_argument('tgt_label', type=int)
    parser.add_argument('--used_column', type=str, default="train")
    parser.add_argument('--start_layer_idx', type=int, default=9)
    args = parser.parse_args()
    tgt_label = args.tgt_label
    start_layer_idx = args.start_layer_idx
    used_column = args.used_column
    # argparseで受け取った引数のサマリーを表示
    print(f"tgt_label: {tgt_label}, start_layer_idx: {start_layer_idx}, used_column: {used_column}")

    # デバイス (cuda, or cpu) の取得
    device = get_device()
    # datasetをロード (初回の読み込みだけやや時間かかる)
    cifar10 = load_from_disk(os.path.join(ViTExperiment.DATASET_DIR, "c10"))
    # 指定したラベルだけ集めたデータセット
    tgt_dataset = cifar10[used_column].filter(lambda x: x["label"] == tgt_label)
    # 読み込まれた時にリアルタイムで前処理を適用するようにする
    cifar10_preprocessed = tgt_dataset.with_transform(transforms)
    # ラベルを示す文字列のlist
    labels = cifar10_preprocessed.features["label"].names
    # pretrained modelのロード
    pretrained_dir = ViTExperiment.OUTPUT_DIR
    model = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
    model.eval()


    # ひとまず適当にtgt_layerを決めて中間層のニューロンを取得する
    tgt_pos = ViTExperiment.CLS_IDX # tgt_posはCLS_IDXで固定 (intermediate_statesの2次元目の0番目の要素に対応する中間層ニューロン)
    num_points = ViTExperiment.NUM_POINTS # integrated gradientの積分近似の分割数
    end_layer_idx = model.vit.config.num_hidden_layers
    
    # record various results
    res_dict = {
        'pred': [],
        'base': [],
        'ig_list': []
    }
    cache_dir = os.path.join(ViTExperiment.OUTPUT_DIR, f"cache_states_{used_column}")
    tic = time.perf_counter()

    # loop for the layer
    for tgt_layer in range(start_layer_idx, end_layer_idx):
        print(f"tgt_layer={tgt_layer}")
        grad_list, base_list = [], []
        # 対象のレイヤのintermediate neuronsの値を取得
        intermediate_save_path = os.path.join(cache_dir, f"intermediate_states_l{tgt_layer}.pt")
        cached_mid_states = torch.load(intermediate_save_path, map_location="cpu")

        # loop for the dataset
        for data_idx, entry_dic in tqdm(enumerate(cifar10_preprocessed.iter(batch_size=1)), 
                                total=len(cifar10_preprocessed)): # NOTE: 重みを変える分でバッチ次元使ってるのでデータサンプルにバッチ次元をできない (データのバッチ化ができない)
            # if data_idx == 100:
            #     break
            # data_idxに対応するcached statesを取得
            tgt_mid = torch.unsqueeze(cached_mid_states[data_idx], 0).to(device)
            # data取得
            x, y = entry_dic["pixel_values"].to(device), entry_dic["labels"][0]
            # get scaled weights
            scaled_weights, weights_step = scaled_input(tgt_mid, num_points)  # (num_points, ffn_size), (ffn_size)
            scaled_weights.requires_grad_(True)
            output = model(x, tgt_pos=tgt_pos, tgt_layer=tgt_layer, tmp_score=scaled_weights, tgt_label=y)
            grad = output.gradients
            # this var stores the partial diff. for each scaled weights
            grad = grad.sum(dim=0)  # (ffn_size) # ここが積分計算の近似値
            grad_list.append(grad.tolist())
            base_list.append(tgt_mid.squeeze().tolist())
        res_dict['ig_list'].append(np.array(grad_list))
        res_dict['base'].append(np.array(base_list))
    
    res_dict['ig_list'] = np.array(res_dict['ig_list']).transpose(1, 0, 2)
    res_dict['base'] = np.array(res_dict['base']).transpose(1, 0, 2)
    print(res_dict['ig_list'].shape) # (num_sample, num_tgt_layers, 3072) = (サンプル数, 対象レイヤ数, 中間ニューロン数)
    print(res_dict['base'].shape) # (num_sample, num_tgt_layers, 3072) = (サンプル数, 対象レイヤ数, 中間ニューロン数)
    # result_dirがなかったら作る
    result_dir = os.path.join(ViTExperiment.OUTPUT_DIR, "neuron_scores")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # npyで三次元配列を保存
    ig_list_save_path = os.path.join(result_dir, f"ig_list_l{start_layer_idx}tol{model.vit.config.num_hidden_layers}_{tgt_label}.npy")
    np.save(ig_list_save_path, res_dict['ig_list'])
    base_save_path = os.path.join(result_dir, f"base_l{start_layer_idx}tol{model.vit.config.num_hidden_layers}_{tgt_label}.npy")
    np.save(base_save_path, res_dict['base'])
    print(f"ig_list: {res_dict['ig_list'].shape} is saved at {ig_list_save_path}.")
    print(f"base: {res_dict['base'].shape} is saved at {base_save_path}.")

    toc = time.perf_counter()
    print(f"***** Costing time: {toc - tic:0.4f} seconds *****")
