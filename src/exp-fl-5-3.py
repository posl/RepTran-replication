import os, sys, time, pickle
import argparse
from tqdm import tqdm
import numpy as np
import torch
from datasets import load_from_disk
from transformers import ViTForImageClassification
from utils.helper import get_device
from utils.vit_util import transforms, transforms_c100, ViTFromLastLayer
from utils.constant import ViTExperiment
from utils.log import set_exp_logging
from logging import getLogger

logger = getLogger("base_logger")


if __name__ == "__main__":
    # データセットをargparseで受け取る
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str)
    parser.add_argument('k', type=int, help="the fold id (0 to K-1)")
    # cache処理をするかどうかのフラグ
    parser.add_argument('--no_cache', action='store_true', help="whether to cache hidden states before layernorm or not")
    parser.add_argument("--tgt_split", type=str, default="repair", help="the target split for the cache")
    args = parser.parse_args()
    ds_name = args.ds
    k = args.k
    tgt_split = args.tgt_split
    do_not_cache = args.no_cache # default: False
    pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR.format(k=k)
    # このpythonのファイル名を取得
    this_file_name = os.path.basename(__file__).split(".")[0]
    logger = set_exp_logging(exp_dir=pretrained_dir, exp_name=this_file_name)
    logger.info(f"ds_name: {ds_name}, fold_id: {k}")

    # datasetごとに違う変数のセット
    if ds_name == "c10":
        tf_func = transforms
        label_col = "label"
    elif ds_name == "c100":
        tf_func = transforms_c100
        label_col = "fine_label"
    else:
        NotImplementedError
    tgt_pos = ViTExperiment.CLS_IDX
    ds_dirname = f"{ds_name}_fold{k}"
    # デバイス (cuda, or cpu) の取得
    device = get_device()
    # datasetをロード (初回の読み込みだけやや時間かかる)
    ds = load_from_disk(os.path.join(ViTExperiment.DATASET_DIR, ds_dirname))
    # ラベルの取得
    labels = {
        "train": np.array(ds["train"][label_col]),
        "repair": np.array(ds["repair"][label_col]),
        "test": np.array(ds["test"][label_col])
    }
    # 読み込まれた時にリアルタイムで前処理を適用するようにする
    ds_preprocessed = ds.with_transform(tf_func)
    # 1エポック目のモデルロード
    model = ViTForImageClassification.from_pretrained(os.path.join(pretrained_dir, "checkpoint-1250"))
    model.to(device)
    model.eval()
    # configuration
    end_li = model.vit.config.num_hidden_layers
    batch_size = ViTExperiment.BATCH_SIZE
    tgt_ds = ds_preprocessed[tgt_split]
    tgt_labels = labels[tgt_split]
    tgt_layer = 11 # NOTE: we only use the last layer for repairing
    print(f"tgt_layer: {tgt_layer}, tgt_split: {tgt_split}")
    logger.info(f"tgt_layer: {tgt_layer}, tgt_split: {tgt_split}")

    # キャッシュの保存用のディレクトリ
    exp_dir = f"/src/src/exp-fl-5/{ds_name}_fold{k}"
    # hs_bef_norm_dir
    hs_bef_norm_dir = os.path.join(exp_dir, f"cache_hidden_states_before_layernorm_{tgt_split}")
    os.makedirs(hs_bef_norm_dir, exist_ok=True)
    hs_bef_norm_path = os.path.join(hs_bef_norm_dir, f"hidden_states_before_layernorm_{tgt_layer}.npy")
    # intermediate_state
    mid_dir = os.path.join(exp_dir, f"cache_states_{tgt_split}")
    os.makedirs(mid_dir, exist_ok=True)
    mid_path = os.path.join(mid_dir, f"intermediate_states_l{tgt_layer}.npy")
    
    if not do_not_cache:
        logger.info(f"starting the caching process.")
        # tgt_splitのサンプルを入力してlayernormの前の隠れ状態を取得
        all_logits = []
        all_hidden_states_before_layernorm = []
        all_intermediate_states = []
        ts_without_cache = time.perf_counter() # 開始時刻
        for entry_dic in tqdm(ds_preprocessed[tgt_split].iter(batch_size=batch_size), total=len(ds_preprocessed[tgt_split])//batch_size+1):
            x, y = entry_dic["pixel_values"].to(device), entry_dic["labels"]
            output = model.forward(x, output_intermediate_states=True, output_hidden_states_before_layernorm=True, tgt_pos=tgt_pos)
            logits = output.logits.detach().cpu().numpy()
            # FFN直前のlayernormの前の隠れ状態を取得
            hidden_states_before_layernorm = output.hidden_states_before_layernorm[tgt_layer].detach().cpu().numpy()
            # print(hidden_states_before_layernorm.shape) # (batch_size, seq_len, num_neurons_hbef)
            # 中間層の隠れ状態も取得
            intermediate_states = output.intermediate_states[tgt_layer][1]
            # print(intermediate_states.shape) # (batch_size, num_neurons_hmid)
            all_logits.append(logits)
            all_hidden_states_before_layernorm.append(hidden_states_before_layernorm)
            all_intermediate_states.append(intermediate_states)
        all_logits = np.concatenate(all_logits)
        all_pred_labels = all_logits.argmax(axis=-1)
        all_hidden_states_before_layernorm = np.concatenate(all_hidden_states_before_layernorm)
        all_intermediate_states = np.concatenate(all_intermediate_states)
        is_correct = np.equal(all_pred_labels, tgt_labels)
        print(f"{sum(is_correct)} / {len(is_correct)}")
        logger.info(f"{sum(is_correct)} / {len(is_correct)}")
        te_without_cache = time.perf_counter() # 終了時刻
        t_without_cache = te_without_cache - ts_without_cache

        # all_hidden_states_before_layernormはキャッシュとして保存
        np.save(hs_bef_norm_path, all_hidden_states_before_layernorm)
        logger.info(f"all_hidden_states_before_layernorm {all_hidden_states_before_layernorm.shape} has been saved at hs_bef_norm_path: {hs_bef_norm_path}")

        # all_intermediate_statesも保存
        np.save(mid_path, all_intermediate_states)
        logger.info(f"all_intermediate_states {all_intermediate_states.shape} has been saved at mid_path: {mid_path}")
        

    logger.info(f"checking the cached states.")
    assert os.path.exists(hs_bef_norm_path), f"cache_path: {hs_bef_norm_path} does not exist."
    # Check the cached hidden states and ViTFromLastLayer
    cached_hidden_states = np.load(hs_bef_norm_path)
    # ViTFromLastLayerのテスト
    vit_from_last_layer = ViTFromLastLayer(model)
    hidden_states_before_layernorm = torch.from_numpy(cached_hidden_states).to(device)
    # hidden_states_before_layernormを32個ずつにバッチ化
    num_batches = (hidden_states_before_layernorm.shape[0] + batch_size - 1) // batch_size  # バッチの数を計算 (最後の中途半端なバッチも使いたいので，切り上げ)
    batched_hidden_states_before_layernorm = np.array_split(hidden_states_before_layernorm, num_batches)

    all_logits = []
    ts_with_cahce = time.perf_counter()
    for cached_state in tqdm(batched_hidden_states_before_layernorm, total=len(batched_hidden_states_before_layernorm)):
        logits = vit_from_last_layer(hidden_states_before_layernorm=cached_state)
        logits = logits.detach().cpu().numpy()
        all_logits.append(logits)
    all_logits = np.concatenate(all_logits)
    all_pred_labels = all_logits.argmax(axis=-1)
    is_correct = np.equal(all_pred_labels, tgt_labels)
    print(f"{sum(is_correct)} / {len(is_correct)}")
    logger.info(f"{sum(is_correct)} / {len(is_correct)}")
    if not do_not_cache: # cacheした場合は時間の比較を出す
        te_with_cache = time.perf_counter()
        t_with_cache = te_with_cache - ts_with_cahce
        speedup_rate = t_without_cache / t_with_cache * 100
        logger.info(f"t_without_cache: {t_without_cache:.5f} sec., t_with_cache: {t_with_cache:.5f} sec., speedup_rate: {speedup_rate:.5f} %")