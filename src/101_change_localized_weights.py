import os, sys, time, pickle, json, math
import argparse
import torch
from tqdm import tqdm
from collections import defaultdict
from itertools import product
import numpy as np
from utils.helper import get_device, json2dict
from utils.vit_util import localize_weights, localize_weights_random, identfy_tgt_misclf, transforms, transforms_c100, ViTFromLastLayer
from utils.constant import ViTExperiment
from utils.log import set_exp_logging
from utils.de import set_new_weights
from datasets import load_from_disk
from transformers import ViTForImageClassification
from logging import getLogger

logger = getLogger("base_logger")

def main(ds_name, k, tgt_rank, n, fl_method, misclf_type, fpfn, run_all):
    device = get_device()
    print(f"ds_name: {ds_name}, fold_id: {k}, tgt_rank: {tgt_rank}, n: {n}, fl_method: {fl_method}, misclf_type: {misclf_type}, fpfn: {fpfn}")
    # pretrained modelのディレクトリ
    pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR.format(k=k)
    
    # 結果とかログの保存先を先に作っておく
    save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_weights_location")
    if misclf_type == "all":
        save_dir = os.path.join(pretrained_dir, f"all_weights_location")
    if fpfn is not None and misclf_type == "tgt":
        save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_{fpfn}_weights_location")
    location_save_path = os.path.join(save_dir, f"location_n{n}_{fl_method}.npy")
    os.makedirs(save_dir, exist_ok=True)
    # このpythonのファイル名を取得
    this_file_name = os.path.basename(__file__).split(".")[0]
    exp_name = f"{this_file_name}_n{n}" if not run_all else f"{this_file_name}_run_all"
    # loggerの設定をして設定情報を表示
    logger = set_exp_logging(exp_dir=save_dir, exp_name=exp_name)
    logger.info(f"ds_name: {ds_name}, fold_id: {k}, tgt_rank: {tgt_rank}, n: {n}, fl_method: {fl_method}, misclf_type: {misclf_type}")

    # localized position を取り出す
    pos_before, pos_after = np.load(location_save_path)
    # pos_before, pos_afterのshapeを確認
    logger.info(f"pos_before: {pos_before.shape}, pos_after: {pos_after.shape}") # (4n**2, 2), (4n**2, 2)

    # datasetをロード (初回の読み込みだけやや時間かかる)
    ds_dirname = f"{ds_name}_fold{k}"
    ds = load_from_disk(os.path.join(ViTExperiment.DATASET_DIR, ds_dirname))
    tgt_split = "repair"
    tgt_layer = 11
    # datasetごとに違う変数のセット
    if ds_name == "c10" or ds_name == "c10c":
        tf_func = transforms
        label_col = "label"
    elif ds_name == "c100" or ds_name == "c10/0c":
        tf_func = transforms_c100
        label_col = "fine_label"
    else:
        NotImplementedError
    true_labels = ds[tgt_split][label_col]

    # キャッシュの保存用のディレクトリ
    cache_dir = os.path.join(pretrained_dir, f"cache_hidden_states_before_layernorm_{tgt_split}")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"hidden_states_before_layernorm_{tgt_layer}.npy")
    # cache_pathに存在することを確認
    assert os.path.exists(cache_path), f"cache_path: {cache_path} does not exist."
    # Check the cached hidden states and ViTFromLastLayer
    cached_hidden_states = np.load(cache_path)
    logger.info(f"cached_hidden_states: {cached_hidden_states.shape}")
    hidden_states_before_layernorm = torch.from_numpy(cached_hidden_states).to(device)
    batch_size = ViTExperiment.BATCH_SIZE
    num_batches = (hidden_states_before_layernorm.shape[0] + batch_size - 1) // batch_size  # バッチの数を計算 (最後の中途半端なバッチも使いたいので，切り上げ)
    batched_hidden_states_before_layernorm = np.array_split(hidden_states_before_layernorm, num_batches)

    for op in ["enh", "sup"]:
        # 学習済みモデルのロード
        model = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
        model.eval()
        # ViTFromLastLayerのインスタンスを作成
        vit_from_last_layer = ViTFromLastLayer(model)
        if op is not None:
            # モデルの重みを変更する
            dummy_in = [0] * (len(pos_before) + len(pos_after))
            set_new_weights(dummy_in, pos_before, pos_after, vit_from_last_layer, op=op)
        all_logits = []
        all_proba = []
        for cached_state in tqdm(batched_hidden_states_before_layernorm, total=len(batched_hidden_states_before_layernorm)):
            logits = vit_from_last_layer(hidden_states_before_layernorm=cached_state)
            proba = torch.nn.functional.softmax(logits, dim=-1)
            logits = logits.detach().cpu().numpy()
            proba = proba.detach().cpu().numpy()
            all_logits.append(logits)
            all_proba.append(proba)
        all_logits = np.concatenate(all_logits, axis=0)
        all_proba = np.concatenate(all_proba, axis=0)
        all_pred_labels = all_logits.argmax(axis=-1)
        logger.info(f"all_logits: {all_logits.shape}, all_proba: {all_proba.shape}, all_pred_labels: {all_pred_labels.shape}")
        is_correct = np.equal(all_pred_labels, true_labels)
        logger.info(f"{sum(is_correct)} / {len(is_correct)}")

        # true_pred_labelsの値ごとにprobaを取り出す
        proba_dict = defaultdict(list)
        for true_label, proba in zip(true_labels, all_proba):
            proba_dict[true_label].append(proba)
        for true_label, proba_list in proba_dict.items():
            proba_dict[true_label] = np.stack(proba_list)
        for true_label, proba in proba_dict.items():
            logger.info(f"true_label: {true_label}, proba: {proba.shape}")
            save_path = os.path.join(save_dir, f"{tgt_split}_proba_{fl_method}_{op}_{true_label}.npy")
            np.save(save_path, proba)
            logger.info(f"proba: {proba.shape} is saved at {save_path}")

if __name__ == "__main__":
    # データセットをargparseで受け取る
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str)
    parser.add_argument('k', nargs="?", type=list, help="the fold id (0 to K-1)")
    parser.add_argument('tgt_rank', nargs="?", type=list, help="the rank of the target misclassification type")
    parser.add_argument('n', nargs="?", type=int, help="the factor for the number of neurons to fix")
    parser.add_argument('--misclf_type', type=str, help="the type of misclassification (src_tgt or tgt)", default="tgt")
    parser.add_argument("--fpfn", type=str, help="the type of misclassification (fp or fn)", default=None, choices=["fp", "fn"])
    parser.add_argument("--fl_method", type=str, help="the method used for FL", default="vdiff")
    parser.add_argument("--run_all", action="store_true", help="run all settings")
    args = parser.parse_args()
    ds = args.ds
    k_list = args.k
    tgt_rank_list = args.tgt_rank
    n_list = args.n
    misclf_type = args.misclf_type
    fpfn = args.fpfn
    fl_method = args.fl_method
    run_all = args.run_all
    
    assert fl_method == "vdiff" or fl_method == "random", "fl_method should be vdiff or random."

    if run_all:
        # run_allがtrueなのにkとtgt_rankが指定されている場合はエラー
        assert k_list is None and tgt_rank_list is None and n_list is None, "run_all and k_list or tgt_rank_list or n_list cannot be specified at the same time"
        k_list = range(5)
        tgt_rank_list = range(1, 6)
        n_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 77, 109]
        fl_method_list = ["vdiff", "random"]
        misclf_type_list = ["all", "src_tgt", "tgt"]
        fpfn_list = [None, "fp", "fn"]
        for k, tgt_rank, n, fl_method, misclf_type, fpfn in product(k_list, tgt_rank_list, n_list, fl_method_list, misclf_type_list, fpfn_list):
            if (misclf_type == "src_tgt" or misclf_type == "all") and fpfn is not None:
                continue
            main(ds, k, tgt_rank, n, fl_method, misclf_type, fpfn, run_all=run_all)
    else:
        assert k_list is not None and tgt_rank_list is not None and n_list is not None, "k_list and tgt_rank_list and n_list should be specified"
        for k, tgt_rank, n in zip(k_list, tgt_rank_list, n_list):
            main(ds, k, tgt_rank, n, fl_method, misclf_type, fpfn)