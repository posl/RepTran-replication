import os, sys, time, pickle, json, math
import torch
from tqdm import tqdm
from collections import defaultdict
from itertools import product
import numpy as np
from utils.helper import get_device, json2dict
from utils.vit_util import ViTFromLastLayer
from utils.de import set_new_weights, check_new_weights
from utils.constant import ViTExperiment, Experiment1
from utils.log import set_exp_logging
from logging import getLogger
from datasets import load_from_disk
from transformers import ViTForImageClassification

logger = getLogger("base_logger")
tgt_pos = ViTExperiment.CLS_IDX
NUM_IDENTIFIED_NEURONS = Experiment1.NUM_IDENTIFIED_NEURONS # exp-fl-1.md参照
NUM_IDENTIFIED_WEIGHTS = Experiment1.NUM_IDENTIFIED_NEURONS # exp-fl-1.md参照

def get_save_dir(pretrained_dir, tgt_rank, misclf_type, fpfn):
    save_dir = os.path.join(
        pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_weights_location"
    )
    if misclf_type == "all":
        save_dir = os.path.join(pretrained_dir, f"all_weights_location")
    if fpfn is not None and misclf_type == "tgt":
        save_dir = os.path.join(
            pretrained_dir,
            f"misclf_top{tgt_rank}",
            f"{misclf_type}_{fpfn}_weights_location",
        )
    return save_dir

def main(ds_name, k, tgt_rank, misclf_type, fpfn, fl_method):
    device = get_device()
    print(f"ds_name: {ds_name}, fold_id: {k}, tgt_rank: {tgt_rank}, misclf_type: {misclf_type}, fpfn: {fpfn}")
    
    # 定数
    tgt_split = "repair" # NOTE: we only use repair split for repairing
    tgt_layer = 11 # NOTE: we only use the last layer for repairing
    
    if fl_method == "ig":
        n = NUM_IDENTIFIED_NEURONS 
        fl_target = "neuron"
    elif fl_method == "bl":
        n = None
        fl_target = "weight"
    
    # datasetをロード (true_labelsが欲しいので)
    ds_dirname = f"{ds_name}_fold{k}"
    ds = load_from_disk(os.path.join(ViTExperiment.DATASET_DIR, ds_dirname))
    label_col = "fine_label"
    true_labels = ds[tgt_split][label_col]
    
    # ニューロンへの介入の方法のリスト
    op_list = ["enhance", "suppress"]
    
    # pretrained modelのディレクトリ
    pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR.format(k=k)
    # 結果とかログの保存先
    save_dir = get_save_dir(pretrained_dir, tgt_rank, misclf_type, fpfn)
    
    # このpythonのファイル名を取得
    this_file_name = os.path.basename(__file__).split(".")[0]
    exp_name = f"exp-fl-2_{this_file_name}"
    # loggerの設定をして設定情報を表示
    logger = set_exp_logging(exp_dir=save_dir, exp_name=exp_name)
    logger.info(f"ds_name: {ds_name}, fold_id: {k}, tgt_rank: {tgt_rank}, n: {n}, misclf_type: {misclf_type}, tgt_split: {tgt_split}, tgt_layer: {tgt_layer}")
    
    # localization結果をロード
    print(f"method_name: {fl_method}")
    if fl_method == "ig": # ig の場合は特定するニューロン数nを設定できる
        location_save_path = os.path.join(save_dir, f"exp-fl-2_location_n{n}_neuron_{fl_method}.npy")
        proba_save_dir = os.path.join(save_dir, f"exp-fl-2_proba_n{n}_{fl_method}")
    elif fl_method == "bl": # bl の場合は特定する重みの数は特定できない (pareto front を取るので)
        location_save_path = os.path.join(save_dir, f"exp-fl-2_location_weight_{fl_method}.npy")
        proba_save_dir = os.path.join(save_dir, f"exp-fl-2_proba_{fl_method}")
    os.makedirs(proba_save_dir, exist_ok=True)
    places_to_fix = np.load(location_save_path, allow_pickle=True)
    # check for places_to_fix
    if fl_target == "neuron":
        print(f"places_to_fix.shape: {places_to_fix.shape}")
    elif fl_target == "weight":
        print(f"len(places_to_fix): {len(places_to_fix)}")
        if len(places_to_fix) == 2:
            pos_before, pos_after = places_to_fix
            print(f"pos_before.shape: {pos_before.shape}, pos_after.shape: {pos_after.shape}")
    
    # ==============================================================
    # 読み込んだニューロン/重みの位置情報から，モデルに介入を加える
    # ==============================================================
    # キャッシュの保存用のディレクトリ
    cache_dir = os.path.join(pretrained_dir, f"cache_hidden_states_before_layernorm_{tgt_split}")
    cache_path = os.path.join(cache_dir, f"hidden_states_before_layernorm_{tgt_layer}.npy")
    # cache_pathに存在することを確認
    assert os.path.exists(cache_path), f"cache_path: {cache_path} does not exist."
    cached_hidden_states = np.load(cache_path)
    hidden_states_before_layernorm = torch.from_numpy(cached_hidden_states).to(device)
    print(f"hidden_states_before_layernorm.shape: {hidden_states_before_layernorm.shape}")
    batch_size = ViTExperiment.BATCH_SIZE
    num_batches = (hidden_states_before_layernorm.shape[0] + batch_size - 1) // batch_size  # バッチの数を計算 (最後の中途半端なバッチも使いたいので，切り上げ)
    batched_hidden_states_before_layernorm = np.array_split(hidden_states_before_layernorm, num_batches) # hidden_states_before_layernormをnum_batches個のsubarrayに分ける
    
    # 2倍にするenh, 0にするsupの操作を繰り返す
    for op in op_list:
        # 学習済みモデルのロード
        model = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
        model.eval()
        # ViTFromLastLayerのインスタンスを作成
        vit_from_last_layer = ViTFromLastLayer(model)
        if fl_target == "neuron":
            print(f"places_to_fix.shape: {places_to_fix.shape}")
            # ニューロン単位での介入は，推論時に動的にやる
            imp_pos, imp_op = places_to_fix, op
        elif fl_target == "weight":
            pos_before, pos_after = places_to_fix
            print(f"pos_before.shape: {pos_before.shape}, pos_after.shape: {pos_after.shape}")
            # 介入を加える (重みを2倍もしくは0倍にする)
            # 重み単位での介入は，推論前に静的にやる（事前に重みを変える）
            dummy_in = [0] * (len(pos_before) + len(pos_after))
            set_new_weights(dummy_in, pos_before, pos_after, vit_from_last_layer, op=op)
            # fl_target == neurons の時用の変数はNoneにしておく
            imp_pos, imp_op = None, None
        
        # 予測の実行
        all_logits = []
        all_proba = []
        for cached_state in tqdm(
            batched_hidden_states_before_layernorm,
            total=len(batched_hidden_states_before_layernorm),
        ):
            # ここでViTFromLastLayerのforwardが実行される
            logits = vit_from_last_layer(
                hidden_states_before_layernorm=cached_state,
                tgt_pos=tgt_pos, imp_pos=imp_pos, imp_op=imp_op
            )
            proba = torch.nn.functional.softmax(logits, dim=-1)
            logits = logits.detach().cpu().numpy()
            proba = proba.detach().cpu().numpy()
            all_logits.append(logits)
            all_proba.append(proba)
        all_logits = np.concatenate(all_logits, axis=0)
        all_proba = np.concatenate(all_proba, axis=0)
        all_pred_labels = all_logits.argmax(axis=-1)

        # true_pred_labelsの値ごとにprobaを取り出す
        proba_dict = defaultdict(list)
        for true_label, proba in zip(true_labels, all_proba):
            proba_dict[true_label].append(proba)
        for true_label, proba_list in proba_dict.items():
            proba_dict[true_label] = np.stack(proba_list)
        for true_label, proba in proba_dict.items():
            save_path = os.path.join(proba_save_dir, f"{tgt_split}_proba_{op}_{true_label}.npy")
            np.save(save_path, proba)
            print(f"saved at {save_path}")
            

if __name__ == "__main__":
    ds = "c100"
    k_list = range(5)
    tgt_rank_list = range(1, 6)
    misclf_type_list = ["all", "src_tgt", "tgt"]
    fpfn_list = [None, "fp", "fn"]
    fl_method_list = ["ig", "bl"]
    for k, tgt_rank, misclf_type, fpfn, fl_method in product(k_list, tgt_rank_list, misclf_type_list, fpfn_list, fl_method_list):
        print(f"k: {k}, tgt_rank: {tgt_rank}, misclf_type: {misclf_type}, fpfn: {fpfn}, fl_method: {fl_method}")
        if (misclf_type == "src_tgt" or misclf_type == "all") and fpfn is not None: # misclf_type == "src_tgt" or "all"の時はfpfnはNoneだけでいい
            continue
        if misclf_type == "all" and tgt_rank != 1: # misclf_type == "all"の時にtgt_rankは関係ないのでこのループもスキップすべき
            continue
        if misclf_type != "all" and fl_method == "ig": # igは誤分類のタイプごとには計算できない
            continue
        main(ds, k, tgt_rank, misclf_type, fpfn, fl_method)
    