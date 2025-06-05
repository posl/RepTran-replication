import os, sys, time, pickle, json
import argparse
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import re
import torch
from datasets import load_from_disk
from transformers import ViTForImageClassification
from utils.helper import get_device, json2dict
from utils.vit_util import transforms, transforms_c100, ViTFromLastLayer, identfy_tgt_misclf, get_ori_model_predictions, get_new_model_predictions, get_batched_hs, get_batched_labels, sample_from_correct_samples, sample_true_positive_indices_per_class, maybe_initialize_repair_weights_
from utils.constant import ViTExperiment, Experiment3
from utils.log import set_exp_logging
from utils.de import DE_searcher
from logging import getLogger
from sklearn.metrics import confusion_matrix

logger = getLogger("base_logger")

DEFAULT_SETTINGS = {
    "n": 5, 
    "ratio_sampled_from_correct": 0.05,  # 正解サンプルのうち何割をrepairに使うか
    "num_sampled_from_correct": 200,  # 正解サンプルのうち何サンプルをrepairに使うか
    "max_search_num": 50,
    "pop_size": 100,
    "alpha": 0.5
}

TGT_SPLIT = "repair"
TGT_LAYER = 11

def gen_and_save_random_location(wnum, model, save_path):
    """
    Generate random locations for the weights to repair and save them.
    """
    bef_shape = model.vit.encoder.layer[TGT_LAYER].intermediate.dense.weight.shape
    aft_shape = model.vit.encoder.layer[TGT_LAYER].output.dense.weight.shape
    
    num_bef = bef_shape[0] * bef_shape[1]
    num_aft = aft_shape[0] * aft_shape[1]
    total = num_bef + num_aft
    assert wnum <= total, f"wnum={wnum} exceeds total number of weights {total}"

    # ランダムにインデックスを選ぶ
    chosen_flat_indices = np.random.choice(total, wnum, replace=False)

    # before/after に分けて元の shape に戻す
    bef_indices = chosen_flat_indices[chosen_flat_indices < num_bef]
    aft_indices = chosen_flat_indices[chosen_flat_indices >= num_bef] - num_bef

    pos_before = np.array([np.unravel_index(i, bef_shape) for i in bef_indices])
    pos_after = np.array([np.unravel_index(i, aft_shape) for i in aft_indices])
    np.save(location_save_path, (pos_before, pos_after))
    print(f"Random locations saved to {location_save_path}.\nlen(pos_before): {len(pos_before)}, len(pos_after): {len(pos_after)}")

def delete_old_files(patch_save_path, tracker_save_path, tgt_indices_save_path, metrics_json_path, wnum):
    # 引数をまとめてイテレーション
    for path in (patch_save_path, tracker_save_path, tgt_indices_save_path, metrics_json_path):
        dirname  = os.path.dirname(path)          # ディレクトリ
        basename = os.path.basename(path)         # ファイル名
        alt_basename = re.sub(rf'_n{wnum}_', '_', basename)
        alt_path     = os.path.join(dirname, alt_basename)
        if not os.path.exists(alt_path):
            print(f"[INFO] {alt_path} does not exist. Skip deleting.")
            continue
        os.remove(alt_path)
        print(f"[INFO] deleted old file: {alt_path}")

if __name__ == "__main__":
    # データセットをargparseで受け取る
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str)
    parser.add_argument('k', type=int, help="the fold id (0 to K-1)")
    parser.add_argument('tgt_rank', type=int, help="the rank of the target misclassification type")
    parser.add_argument('reps_id', type=int, help="the repetition id")
    parser.add_argument("wnum", type=int, help="the number of weights to repair")
    parser.add_argument("--setting_path", type=str, help="path to the setting json file", default=None)
    parser.add_argument("--fl_method", type=str, help="the method used for FL", default="vdiff")
    parser.add_argument('--misclf_type', type=str, help="the type of misclassification (src_tgt or tgt or all)", default="tgt")
    parser.add_argument('--only_eval', action='store_true', help="if True, only evaluate the saved patch", default=False)
    parser.add_argument("--custom_alpha", type=float, help="the custom alpha for the repair", default=None)
    parser.add_argument("--include_other_TP_for_fitness", action="store_true", help="if True, include other TP samples for fitness calculation", default=False)
    parser.add_argument("--fpfn", type=str, help="the type of misclassification (fp or fn)", default=None, choices=["fp", "fn"])
    parser.add_argument("--custom_bounds", type=str, help="the type of bound for the DE search space", default=None, choices=["Arachne", "ContrRep"])
    args = parser.parse_args()
    ds_name = args.ds
    k = args.k
    tgt_rank = args.tgt_rank
    reps_id = args.reps_id
    wnum = args.wnum
    setting_path = args.setting_path
    fl_method = args.fl_method
    misclf_type = args.misclf_type
    only_eval = args.only_eval
    custom_alpha = args.custom_alpha
    include_other_TP_for_fitness = args.include_other_TP_for_fitness
    fpfn = args.fpfn
    custom_bounds = args.custom_bounds
    print(f"ds_name: {ds_name}, k: {k}, tgt_rank: {tgt_rank}, reps_id: {reps_id}, setting_path: {setting_path}, fl_method: {fl_method}, misclf_type: {misclf_type}, only_eval: {only_eval}, custom_alpha: {custom_alpha}, include_other_TP_for_fitness: {include_other_TP_for_fitness}, fpfn: {fpfn}, custom_bounds: {custom_bounds}")

    # 設定のjsonファイルが指定された場合
    if setting_path is not None:
        assert os.path.exists(setting_path), f"{setting_path} does not exist."
        setting_dic = json2dict(setting_path)
        # setting_idは setting_{setting_id}.json というファイル名になる
        setting_id = os.path.basename(setting_path).split(".")[0].split("_")[-1]
    # 設定のjsonファイルが指定されない場合はnとalphaだけカスタムorデフォルトの設定を使う
    else:
        setting_dic = DEFAULT_SETTINGS
        # custom_alpha, custom_boundsが1つでも指定されている場合はいったん空文字にする
        setting_id = "default" if (custom_alpha is None) and (custom_bounds is None) else ""
        parts = []
        if wnum is not None:
            setting_dic["wnum"] = wnum
            parts.append(f"n{wnum}")
        if custom_alpha is not None:
            setting_dic["alpha"] = custom_alpha
            parts.append(f"alpha{custom_alpha}")
        if custom_bounds is not None:
            setting_dic["bounds"] = custom_bounds
            parts.append(f"bounds{custom_bounds}")
        # リストの要素を'_'で連結
        setting_id = "_".join(parts)
    # pretrained modelのディレクトリ
    pretrained_dir = getattr(ViTExperiment, ds_name.replace("-", "_")).OUTPUT_DIR.format(k=k)
    # 結果とかログの保存先を先に作っておく
    # save_dirは, 5種類の誤分類タイプのどれかを一意に表す
    if fpfn is not None and misclf_type == "tgt": # tgt_fp or tgt_fn
        save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_{fpfn}_repair_weight_by_de")
    elif misclf_type == "all": # all
        save_dir = os.path.join(pretrained_dir, f"{misclf_type}_repair_weight_by_de")
    else: # tgt_all or src_tgt
        save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_repair_weight_by_de")
    os.makedirs(save_dir, exist_ok=True)
    
    # repairの成果物の保存ファイル名
    # ==================================================================
    patch_save_path = os.path.join(save_dir, f"exp-repair-4-1-best_patch_{setting_id}_{fl_method}_reps{reps_id}.npy")
    tracker_save_path = os.path.join(save_dir, f"exp-repair-4-1-tracker_{setting_id}_{fl_method}_reps{reps_id}.pkl")
    # repair に使ったデータの tgt_indices を npy で保存
    tgt_indices_save_path = os.path.join(save_dir, f"exp-repair-4-1-tgt_indices_{setting_id}_{fl_method}_reps{reps_id}.npy") # TODO: tgt_indicesの特定にランダム性が入る場合はそれをトラックできるようなファイル名にする必要あり. 同じsetting_id, fl_methodでもrepetitionが異なる場合
    metrics_json_path = os.path.join(save_dir, f"exp-repair-4-1-metrics_for_repair_{setting_id}_{fl_method}_reps{reps_id}.json") # 各設定での実行時間を記録
    # wnumが入ってないverの古いファイルたちを削除する
    # delete_old_files(patch_save_path, tracker_save_path, tgt_indices_save_path, metrics_json_path, wnum)
    # # 上のファイルたちがすでにそべて存在していたらreturn 0
    # if os.path.exists(patch_save_path) and os.path.exists(tracker_save_path) and os.path.exists(tgt_indices_save_path) and os.path.exists(metrics_json_path):
    #     print(f"All results already exists. Skip this experiment.")
    #     exit(0)
    # ==================================================================
    
    # このpythonのファイル名を取得
    this_file_name = os.path.basename(__file__).split(".")[0]
    exp_name = f"{this_file_name}_{setting_id}"
    # loggerの設定をして設定情報を表示
    logger = set_exp_logging(exp_dir=save_dir, exp_name=exp_name)
    logger.info(f"ds_name: {ds_name}, fold_id: {k}, setting_path: {setting_path}")
    logger.info(f"setting_dic (id={setting_id}): {setting_dic}")

    # datasetごとに違う変数のセット
    if ds_name == "c10" or ds_name == "tiny-imagenet":
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
    # ラベルの取得 (shuffleされない)
    labels = {
        "train": np.array(ds["train"][label_col]),
        "repair": np.array(ds["repair"][label_col]),
        "test": np.array(ds["test"][label_col])
    }
    # 読み込まれた時にリアルタイムで前処理を適用するようにする
    ds_preprocessed = ds.with_transform(tf_func)
    # pretrained modelのロード
    model, loading_info = ViTForImageClassification.from_pretrained(pretrained_dir, output_loading_info=True)
    model.to(device).eval()
    model = maybe_initialize_repair_weights_(model, loading_info["missing_keys"])
    end_li = model.vit.config.num_hidden_layers
    batch_size = ViTExperiment.BATCH_SIZE
    tgt_split = TGT_SPLIT # NOTE: we only use repair split for repairing
    ori_tgt_ds = ds_preprocessed[tgt_split]
    ori_tgt_labels = labels[tgt_split]
    tgt_layer = TGT_LAYER # NOTE: we only use the last layer for repairing
    logger.info(f"tgt_layer: {tgt_layer}, tgt_split: {tgt_split}")

    # ===============================================
    # localization phase
    # ===============================================

    if fpfn is not None and misclf_type == "tgt":
        location_save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_{fpfn}_weights_location")
    elif misclf_type == "all":
        location_save_dir = os.path.join(pretrained_dir, f"all_weights_location")
    else:
        location_save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_weights_location")
    # 重みの位置情報の保存ファイル
    if fl_method == "ours" or fl_method == "bl":
        location_filename = f"exp-repair-4-1_location_n{wnum}_weight_{fl_method}.npy"
        location_save_path = os.path.join(location_save_dir, location_filename)
    elif fl_method == "random":
        location_filename = f"exp-repair-4-1_location_n{wnum}_weight_random_reps{reps_id}.npy" # NOTE: randomの時はreps_idをつける(ランダム性の考慮)
        location_save_path = os.path.join(location_save_dir, location_filename)
        if not os.path.exists(location_save_path):
            # ランダム位置を生成してファイルに保存する
            gen_and_save_random_location(wnum, model, save_path=location_save_path)
    assert os.path.exists(location_save_path), f"{location_save_path} does not exist. Please run the localization phase first."
    pos_before, pos_after = np.load(location_save_path, allow_pickle=True)
    # log表示
    logger.info(f"pos_before={pos_before}")
    logger.info(f"pos_after={pos_after}")
    logger.info(f"num(pos_to_fix)=num(pos_before)+num(pos_before)={len(pos_before)}+{len(pos_after)}={len(pos_before)+len(pos_after)}")
    print(f"num(pos_to_fix)=num(pos_before)+num(pos_before)={len(pos_before)}+{len(pos_after)}={len(pos_before)+len(pos_after)}")
    

    # ===============================================
    # Data preparation for repair
    # ===============================================

    # tgt_rankの誤分類情報を取り出す
    misclf_info_dir = os.path.join(pretrained_dir, "misclf_info")
    misclf_pair, tgt_label, tgt_mis_indices = identfy_tgt_misclf(misclf_info_dir, tgt_split=tgt_split, tgt_rank=tgt_rank, misclf_type=misclf_type, fpfn=fpfn)
    indices_to_incorrect = tgt_mis_indices
    # NOTE: indices_to_incorrectからはsampleしなくてよい？今のところ全部使うのでランダム性が入るのはcorrectのsamplingだけ

    # original model の repair setの各サンプルに対する正解/不正解のインデックスを取得
    pred_res_dir = os.path.join(pretrained_dir, "pred_results", "PredictionOutput")
    if misclf_type == "tgt":
        ori_pred_labels, _, indices_to_correct_tgt, _, indices_to_correct_others = get_ori_model_predictions(pred_res_dir, labels, tgt_split=tgt_split, misclf_type=misclf_type, tgt_label=tgt_label)
        # mis_clf=tgtでも全ての正解サンプルを選ぶ
        indices_to_correct = np.sort(np.concatenate([indices_to_correct_tgt, indices_to_correct_others]))
    else:
        ori_pred_labels, _, indices_to_correct = get_ori_model_predictions(pred_res_dir, labels, tgt_split=tgt_split, misclf_type=misclf_type, tgt_label=tgt_label)
    print(f"len(indices_to_correct): {len(indices_to_correct)}, len(indices_to_incorrect): {len(indices_to_incorrect)}")
    
    num_sampled_from_pos = int(setting_dic["ratio_sampled_from_correct"] * len(ori_tgt_labels))
    # 正解サンプルは一般に多すぎるのでラベルごとの分布を考慮してサンプリングする
    sampled_indices_to_correct = sample_true_positive_indices_per_class(
        num_sampled_from_pos,   # 正解サンプルのうち何サンプルをrepairに使うか
        indices_to_correct,
        ori_pred_labels,
    )
    ori_sampled_indices_to_correct, ori_indices_to_incorrect = sampled_indices_to_correct.copy(), indices_to_incorrect.copy()
    # 抽出した正解データと，全不正解データを結合して1つのデータセットにする
    tgt_indices = sampled_indices_to_correct.tolist() + indices_to_incorrect.tolist() # .tolist() は 非破壊的method
    # ここで (正解サンプル) + (不正解サンプル) の順番にしている
    # tgt_indicesは全てユニークな値であることを保証
    assert len(tgt_indices) == len(set(tgt_indices)), f"len(tgt_indices): {len(tgt_indices)}, len(set(tgt_indices)): {len(set(tgt_indices))}"
    logger.info(f"tgt_indices: {tgt_indices} (len: {len(tgt_indices)})")
    print(f"len(tgt_indices): {len(tgt_indices)}")
    # tgt_indicesに対応するデータトラベルを取り出す
    tgt_ds = ori_tgt_ds.select(tgt_indices)
    tgt_labels = ori_tgt_labels[tgt_indices]
    logger.info(f"ori_pred_labels[tgt_indices]: {ori_pred_labels[tgt_indices]} (len: {len(ori_pred_labels[tgt_indices])})")
    logger.info(f"ori_tgt_labels[tgt_indices]: {tgt_labels} (len: {len(tgt_labels)})")
    # print(f"ori_pred_labels[tgt_indices]: {ori_pred_labels[tgt_indices]} (len: {len(ori_pred_labels[tgt_indices])})")
    
    # repair setに対するhidden_states_before_layernormを取得
    hs_save_dir = os.path.join(pretrained_dir, f"cache_hidden_states_before_layernorm_{tgt_split}")
    hs_save_path = os.path.join(hs_save_dir, f"hidden_states_before_layernorm_{tgt_layer}.npy")
    assert os.path.exists(hs_save_path), f"{hs_save_path} does not exist."
    batch_hs_before_layernorm_tgt = get_batched_hs(hs_save_path, batch_size, tgt_indices, device=device)
    batch_labels_tgt = get_batched_labels(ori_tgt_labels, batch_size, tgt_indices)
    # repair set全体 (tgt_indicesを指定しない) のbatchも作成
    batch_hs_before_layernorm = get_batched_hs(hs_save_path, batch_size, device=device)
    batch_labels = get_batched_labels(ori_tgt_labels, batch_size)

    # NOTE: 同じsetting_idでも乱数のシードを考慮していない
    np.save(tgt_indices_save_path, tgt_indices)

    # ===============================================
    # DE search (patch generation)
    # ===============================================

    # 最終層だけのモデルを準備
    vit_from_last_layer = ViTFromLastLayer(model)
    vit_from_last_layer.eval()

    # 読み込んだ直後の状態の学習済みViTに対して，repair set全体およびrepair対象のデータに対する予測結果を確認する
    # repair set全体
    ori_pred_labels, ori_true_labels = get_new_model_predictions(vit_from_last_layer, batch_hs_before_layernorm, batch_labels, tgt_pos=tgt_pos)
    ori_is_correct = ori_pred_labels == ori_true_labels
    # repair対象のデータ
    ori_pred_labels_tgt, ori_true_labels_tgt = get_new_model_predictions(vit_from_last_layer, batch_hs_before_layernorm_tgt, batch_labels_tgt, tgt_pos=tgt_pos)
    ori_is_correct_tgt = ori_pred_labels_tgt == ori_true_labels_tgt
    # ログ表示
    logger.info(f"sum(ori_is_correct), len(ori_is_correct), {sum(ori_is_correct), len(ori_is_correct)}")
    logger.info(f"sum(ori_is_correct_tgt), len(ori_is_correct_tgt), {sum(ori_is_correct_tgt), len(ori_is_correct_tgt)}")

    # pos_before, pos_afterの位置の重みを最適化の変数にする
    linear_before2med = vit_from_last_layer.base_model_last_layer.intermediate.dense # on GPU
    weight_before2med = linear_before2med.weight.cpu().detach().numpy() # on CPU
    linear_med2after = vit_from_last_layer.base_model_last_layer.output.dense # on GPU
    weight_med2after = linear_med2after.weight.cpu().detach().numpy() # on CPU

    # DE_searcherの初期化
    max_search_num = setting_dic["max_search_num"]
    alpha = setting_dic["alpha"] # [0, 1]の値で，fitnessの正しい分類に対する重み: 誤分類に対する重み = 1-alpha: alpha になる
    assert 0 <= alpha <= 1, f"alpha should be in [0, 1]. alpha: {alpha}"
    logger.info(f"alpha of the fitness func.: {alpha}")
    pop_size = setting_dic["pop_size"]
    num_labels = len(set(labels["train"]))
    # correct, incorrect indicesを更新
    indices_to_correct_for_de_searcher = np.arange(len(sampled_indices_to_correct))
    indices_to_incorrect_for_de_searcher = np.arange(len(sampled_indices_to_correct), len(sampled_indices_to_correct) + len(indices_to_incorrect))
    logger.info(f"indices_to_correct_for_de_searcher: {indices_to_correct_for_de_searcher}, indices_to_incorrect_for_de_searcher: {indices_to_incorrect_for_de_searcher}")
    # batch_hs_before_layernorm_tgtの最初のindices_to_correct_for_de_searcher個に対してはモデルが正解して，残りのindices_to_incorrect_for_de_searcher個に対してはモデルが不正解していることを確認
    assert np.sum(ori_is_correct_tgt[indices_to_correct_for_de_searcher]) == len(indices_to_correct_for_de_searcher), f"ori_is_correct_tgt[indices_to_correct_for_de_searcher]: {ori_is_correct_tgt[indices_to_correct_for_de_searcher]}"
    assert np.sum(ori_is_correct_tgt[indices_to_incorrect_for_de_searcher]) == 0, f"ori_is_correct_tgt[indices_to_incorrect_for_de_searcher]: {ori_is_correct_tgt[indices_to_incorrect_for_de_searcher]}"
    
    # logger.info("Before DE search...")
    # DE_searcherの初期化
    searcher = DE_searcher(
        batch_hs_before_layernorm=batch_hs_before_layernorm_tgt,
        batch_labels=batch_labels_tgt,
        indices_to_correct=indices_to_correct_for_de_searcher,
        indices_to_wrong=indices_to_incorrect_for_de_searcher,
        num_label=num_labels,
        indices_to_target_layers=[tgt_layer],
        device=device,
        mutation=(0.5, 1),
        recombination=0.7,
        max_search_num=max_search_num,
        partial_model=vit_from_last_layer,
        alpha=alpha,
        pop_size=pop_size,
        mode="weight",
        pos_before=pos_before,
        pos_after=pos_after,
        weight_before2med=weight_before2med,
        weight_med2after=weight_med2after,
        custom_bounds=custom_bounds
    )
    logger.info(f"Start DE search...")
    s = time.perf_counter()
    patch, fitness_tracker = searcher.search(patch_save_path=patch_save_path, pos_before=pos_before, pos_after=pos_after, tracker_save_path=tracker_save_path)
    e = time.perf_counter()
    # print(f"[after DE] {vit_from_last_layer.base_model_last_layer.intermediate.dense.weight.data[pos_before[0][0]][pos_before[0][1]]}")
    tot_time = e - s
    logger.info(f"Total execution time: {tot_time} sec.")
    print(f"Total execution time: {tot_time} sec.")
    # 実行時間だけをメトリクスとしてjsonに保存
    # (このjsonはあとでrepair rateなども追記される (007f))
    metrics = {"tot_time": tot_time}
    with open(metrics_json_path, "w") as f:
        json.dump(metrics, f)