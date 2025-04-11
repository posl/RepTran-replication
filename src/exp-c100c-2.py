import os, sys, time, pickle
import argparse
from tqdm import tqdm
import numpy as np
import torch
from datasets import load_from_disk
from transformers import ViTForImageClassification
from utils.helper import get_device, get_bottom3_keys_from_json
from utils.vit_util import transforms, transforms_c100, ViTFromLastLayer
from utils.constant import ViTExperiment, Experiment
from utils.log import set_exp_logging
from logging import getLogger

logger = getLogger("base_logger")


if __name__ == "__main__":
    pretrained_dir = ViTExperiment.c100.OUTPUT_DIR.format(k=0)
    # デバイス (cuda, or cpu) の取得
    device = get_device()
    # このpythonのファイル名を取得
    this_file_name = os.path.basename(__file__).split(".")[0]
    logger = set_exp_logging(exp_dir=pretrained_dir, exp_name=this_file_name)
    label_col = "fine_label"
    tgt_pos = ViTExperiment.CLS_IDX
    
    # Load CIFAR-100-C dataset
    dataset_dir = Experiment.DATASET_DIR
    ds = load_from_disk(os.path.join(dataset_dir, "c100c"))
    labels = {
        key: np.array(ds[key][label_col]) for key in ds.keys()
    }
    ds_preprocessed = ds.with_transform(transforms_c100)
    
    # pretrained modelのロード
    model = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
    model.eval()
    # configuration
    batch_size = ViTExperiment.BATCH_SIZE
    end_li = model.vit.config.num_hidden_layers
    
    # accuracyのbottom3のノイズタイプのみ処理したい
    bottom3_keys = get_bottom3_keys_from_json(os.path.join(pretrained_dir, "corruption_accuracy.json"))
    
    # tqdmで進捗表示しながら処理
    for key in tqdm(ds_preprocessed.keys(), desc="Evaluating corruptions"):
        if key not in bottom3_keys:
            print(f"Skipping {key}...")
            continue
        print(f"Processing {key}...")
        tgt_ds = ds_preprocessed[key]
        tgt_labels = labels[key]
        tgt_layer = 11 # NOTE: we only use the last layer for repairing

        # キャッシュの保存用のディレクトリ
        cache_dir = os.path.join(pretrained_dir, f"cache_hidden_states_before_layernorm_{key}")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"hidden_states_before_layernorm_{tgt_layer}.npy")
        
        # ====================================
        # starting the caching process.
        # ====================================
        logger.info(f"starting the caching process.")
        # tgt_splitのサンプルを入力してlayernormの前の隠れ状態を取得
        all_logits = []
        all_hidden_states_before_layernorm = []
        ts_without_cache = time.perf_counter()
        for entry_dic in tqdm(ds_preprocessed[key].iter(batch_size=batch_size), total=len(ds_preprocessed[key])//batch_size+1):
            x, y = entry_dic["pixel_values"].to(device), entry_dic["labels"]
            output = model.forward(x, output_hidden_states_before_layernorm=True)
            logits = output.logits.detach().cpu().numpy()
            hidden_states_before_layernorm = output.hidden_states_before_layernorm[tgt_layer].detach().cpu().numpy()
            all_logits.append(logits)
            all_hidden_states_before_layernorm.append(hidden_states_before_layernorm)
        all_logits = np.concatenate(all_logits)
        all_pred_labels = all_logits.argmax(axis=-1)
        all_hidden_states_before_layernorm = np.concatenate(all_hidden_states_before_layernorm)
        is_correct = np.equal(all_pred_labels, tgt_labels)
        print(f"{sum(is_correct)} / {len(is_correct)}")
        logger.info(f"{sum(is_correct)} / {len(is_correct)}")
        te_without_cache = time.perf_counter()
        t_without_cache = te_without_cache - ts_without_cache

        # all_hidden_states_before_layernormはキャッシュとして保存
        np.save(cache_path, all_hidden_states_before_layernorm)
        logger.info(f"all_hidden_states_before_layernorm {all_hidden_states_before_layernorm.shape} has been saved at cache_path: {cache_path}")
        
        # ====================================
        # checking the cached states.
        # ====================================
        logger.info(f"checking the cached states.")
        assert os.path.exists(cache_path), f"cache_path: {cache_path} does not exist."
        # Check the cached hidden states and ViTFromLastLayer
        cached_hidden_states = np.load(cache_path, mmap_mode="r")
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
        te_with_cache = time.perf_counter()
        t_with_cache = te_with_cache - ts_with_cahce
        speedup_rate = t_without_cache / t_with_cache * 100
        logger.info(f"t_without_cache: {t_without_cache:.5f} sec., t_with_cache: {t_with_cache:.5f} sec., speedup_rate: {speedup_rate:.5f} %")