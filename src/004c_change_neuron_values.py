import os, sys, time, pickle
import argparse
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import torch
from datasets import load_from_disk
from transformers import ViTForImageClassification
from utils.helper import get_device
from utils.vit_util import transforms, transforms_c100, ViTFromLastLayer
from utils.constant import ViTExperiment
from utils.log import set_exp_logging
from utils.de import DE_searcher
from logging import getLogger

logger = getLogger("base_logger")

def generate_random_positions(start_layer_idx, end_layer_idx, num_neurons, num_kn):
    """
    ランダムに知識ニューロンの位置 (start_layer_idx以上かつend_layer_idx-1以下のレイヤ番号, 0以上num_neurons以下のニューロン番号) を選ぶ
    """
    kn_list = []
    for _ in range(num_kn):
        layer_idx = np.random.randint(start_layer_idx, end_layer_idx)
        neuron_idx = np.random.randint(num_neurons)
        kn_list.append([layer_idx, neuron_idx])
    return kn_list

if __name__ == "__main__":
    # データセットをargparseで受け取る
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str)
    parser.add_argument('k', type=int, help="the fold id (0 to K-1)")
    args = parser.parse_args()
    ds_name = args.ds
    k = args.k
    pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR.format(k=k)
    # このpythonのファイル名を取得
    this_file_name = os.path.basename(__file__).split(".")[0]
    logger = set_exp_logging(exp_dir=pretrained_dir, exp_name=this_file_name)

    print(f"ds_name: {ds_name}, fold_id: {k}")
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
    # pretrained modelのロード
    model = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
    model.eval()
    end_li = model.vit.config.num_hidden_layers
    batch_size = ViTExperiment.BATCH_SIZE
    tgt_split = "repair" # NOTE: we only use repair split for repairing
    tgt_ds = ds_preprocessed[tgt_split]
    tgt_labels = labels[tgt_split]
    tgt_layer = 11 # NOTE: we only use the last layer for repairing
    print(f"tgt_layer: {tgt_layer}, tgt_split: {tgt_split}")
    logger.info(f"tgt_layer: {tgt_layer}, tgt_split: {tgt_split}")

    # ===============================================
    # localization phase
    # ===============================================

    vscore_dir = os.path.join(pretrained_dir, "vscores")
    vmap_dic = defaultdict(defaultdict)
    # 正解と不正解時のvscoreを読み込む
    for cor_mis in ["cor", "mis"]:
        vmap_dic[cor_mis] = defaultdict(np.array)
        ds_type = f"ori_{tgt_split}"
        vscore_save_path = os.path.join(vscore_dir, f"vscore_l1tol{end_li}_all_label_{ds_type}_{cor_mis}.npy")
        vscores = np.load(vscore_save_path)
        vmap_dic[cor_mis] = vscores.T
        print(f"vscores shape ({cor_mis}): {vmap_dic[cor_mis].shape}")
    # take diff of vscores and get top 10% neurons
    vmap_cor = vmap_dic["cor"]
    vmap_mis = vmap_dic["mis"]
    vmap_diff = vmap_cor - vmap_mis # (num_neurons, num_layers)
    # vdiffの上位theta%のニューロンを取得
    theta = 3
    top_theta = np.percentile(np.abs(vmap_diff[:, tgt_layer]), 100-theta)
    condition = np.abs(vmap_diff[:, tgt_layer]).reshape(-1) > top_theta
    logger.info(f"localized_neurons={np.where(condition)[0]}")
    logger.info(f"num(location)={np.where(condition)[0]}")
    places_to_fix = [[tgt_layer, pos] for pos in np.where(condition)[0]]
    # vmap_diff[:, tgt_layer]からconditionに合うものだけ取り出す
    tgt_vdiff = vmap_diff[condition, tgt_layer]

    # ===============================================
    # patch generation phase
    # ===============================================

    # repair setの各サンプルに対して正解/不正解のインデックスを取得
    pred_res_dir = os.path.join(pretrained_dir, "pred_results", "PredictionOutput")
    with open(os.path.join(pred_res_dir, f"{tgt_split}_pred.pkl"), "rb") as f:
        pred_res = pickle.load(f)
    pred_logits = pred_res.predictions
    pred_labels = np.argmax(pred_logits, axis=-1)
    is_correct = pred_labels == labels[tgt_split]
    indices_to_correct = np.where(is_correct)[0]
    indices_to_incorrect = np.where(~is_correct)[0]
    logger.info(f"len(indices_to_correct): {len(indices_to_correct)}, len(indices_to_incorrect): {len(indices_to_incorrect)}")

    # 正解データからrepairに使う一定数だけランダムに取り出す
    num_sampled_from_correct = 200
    indices_to_correct = np.random.choice(indices_to_correct, num_sampled_from_correct, replace=False)
    # 抽出した正解データと，全不正解データを結合して1つのデータセットにする
    tgt_ds = tgt_ds.select(indices_to_correct.tolist() + indices_to_incorrect.tolist())
    tgt_labels = tgt_labels[indices_to_correct.tolist() + indices_to_incorrect.tolist()]
    
    # repair setに対するhidden_states_before_layernormを取得
    hs_save_dir = os.path.join(pretrained_dir, f"cache_hidden_states_before_layernorm_{tgt_split}")
    hs_save_path = os.path.join(hs_save_dir, f"hidden_states_before_layernorm_{tgt_layer}.npy")
    assert os.path.exists(hs_save_path), f"{hs_save_path} does not exist."
    hs_before_layernorm = torch.from_numpy(np.load(hs_save_path)).to(device)
    logger.info(f"hs_before_layernorm is loaded. shape: {hs_before_layernorm.shape}")
    # 使うインデックスに対する状態だけを取り出す
    hs_before_layernorm = hs_before_layernorm[indices_to_correct.tolist() + indices_to_incorrect.tolist()]
    logger.info(f"hs_before_layernorm is sliced. shape: {hs_before_layernorm.shape}")
    # hidden_states_before_layernormを32個ずつにバッチ化
    num_batches = (hs_before_layernorm.shape[0] + batch_size - 1) // batch_size  # バッチの数を計算 (最後の中途半端なバッチも使いたいので，切り上げ)
    batch_hs_before_layernorm = np.array_split(hs_before_layernorm, num_batches)
    
    # correct, incorrect indicesを更新
    indices_to_correct = np.arange(num_sampled_from_correct)
    indices_to_incorrect = np.arange(num_sampled_from_correct, num_sampled_from_correct + len(indices_to_incorrect))
    logger.info(f"len(indices_to_correct): {len(indices_to_correct)}, len(indices_to_incorrect): {len(indices_to_incorrect)}")
    logger.info(f"len(tgt_ds): {len(tgt_ds)}, len(tgt_labels): {len(tgt_labels)}")

    # 最終層だけのモデルを準備
    vit_from_last_layer = ViTFromLastLayer(model)

    # DE_searcherの初期化
    max_search_num = 50
    patch_aggr = 1
    num_labels = len(set(labels["train"]))
    searcher = DE_searcher(
        batch_hs_before_layernorm=batch_hs_before_layernorm,
        labels=tgt_labels,
        indices_to_correct=indices_to_correct,
        indices_to_wrong=indices_to_incorrect,
        num_label=num_labels,
        indices_to_target_layers=[tgt_layer],
        device=device,
        mutation=(0.5, 1),
        recombination=0.7,
        max_search_num=max_search_num,
        partial_model=vit_from_last_layer,
        batch_size=batch_size,
        patch_aggr=patch_aggr,
        tgt_vdiff=tgt_vdiff,
        pop_size=100,
    )

    save_dir = os.path.join(pretrained_dir, "repairing_by_de")
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Start DE search...")
    s = time.perf_counter()
    searcher.search(places_to_fix, save_dir=save_dir)
    e = time.perf_counter()
    logger.info(f"Total execution time: {e-s} sec.")