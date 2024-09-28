import os, sys, time, pickle, json
import argparse
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import evaluate
import torch
from datasets import load_from_disk
from transformers import ViTForImageClassification
from utils.helper import get_device, json2dict
from utils.vit_util import transforms, transforms_c100, ViTFromLastLayer, localize_weights, localize_weights_random, src_tgt_selection, tgt_selection, identfy_tgt_misclf
from utils.data_util import make_batch_of_label
from utils.constant import ViTExperiment
from utils.log import set_exp_logging
from utils.de import DE_searcher
from logging import getLogger
from sklearn.metrics import confusion_matrix

logger = getLogger("base_logger")

DEFAULT_SETTINGS = {
    "n": 5, 
    "num_sampled_from_correct": 200,
    "max_search_num": 50,
    "pop_size": 100,
    "alpha": 0.5
}

def get_ori_model_predictions(pred_res_dir, labels, tgt_split="repair", misclf_type="tgt", tgt_label=None):
    # original model の repair setの各サンプルに対する正解/不正解のインデックスを取得
    pred_res_dir = os.path.join(pretrained_dir, "pred_results", "PredictionOutput")
    with open(os.path.join(pred_res_dir, f"{tgt_split}_pred.pkl"), "rb") as f:
        pred_res = pickle.load(f)
    pred_logits = pred_res.predictions
    ori_pred_labels = np.argmax(pred_logits, axis=-1)
    is_correct = ori_pred_labels == labels[tgt_split]
    # misclf_type == "tgt"の場合は，tgt_labelで正解したものだけをcorrectとして扱う
    if misclf_type == "tgt":
        assert tgt_label is not None, f"tgt_label should be specified when misclf_type is tgt."
        is_correct = is_correct & (labels[tgt_split] == tgt_label)
    indices_to_correct = np.where(is_correct)[0]
    return ori_pred_labels, is_correct, indices_to_correct

def sample_from_correct_samples(num_sampled_from_correct, indices_to_correct):
    if num_sampled_from_correct < len(indices_to_correct):
        sampled_indices_to_correct = np.random.choice(indices_to_correct, num_sampled_from_correct, replace=False)
    else:
        sampled_indices_to_correct = indices_to_correct
    return sampled_indices_to_correct

def get_batched_hs(hs_save_path, batch_size, tgt_indices=None, device=torch.device("cuda")):
    hs_before_layernorm = torch.from_numpy(np.load(hs_save_path)).to(device)
    logger.info(f"hs_before_layernorm is loaded. shape: {hs_before_layernorm.shape}")
    if tgt_indices is not None:
        # 使うインデックスに対する状態だけを取り出す
        hs_before_layernorm_tgt = hs_before_layernorm[tgt_indices]
    else:
        hs_before_layernorm_tgt = hs_before_layernorm
    logger.info(f"hs_before_layernorm is sliced. shape: {hs_before_layernorm_tgt.shape}")
    num_batches = (hs_before_layernorm_tgt.shape[0] + batch_size - 1) // batch_size  # バッチの数を計算 (最後の中途半端なバッチも使いたいので，切り上げ)
    batch_hs_before_layernorm_tgt = np.array_split(hs_before_layernorm_tgt, num_batches)
    return batch_hs_before_layernorm_tgt

def get_batched_labels(labels, batch_size, tgt_indices=None):
    logger.info(f"labels.shape: {labels.shape}")
    if tgt_indices is not None:
        labels_tgt = labels[tgt_indices]
    else:
        labels_tgt = labels
    logger.info(f"labels is sliced. shape: {labels_tgt.shape}")
    num_batches = (len(labels_tgt) + batch_size - 1) // batch_size  # バッチの数を計算 (最後の中途半端なバッチも使いたいので，切り上げ)
    batch_labels = np.array_split(labels_tgt, num_batches)
    return batch_labels

def get_new_model_predictions(vit_from_last_layer, batch_hs_before_layernorm, batch_labels, tgt_pos=0):
    all_pred_labels = []
    all_true_labels = []
    for cache_state, y in zip(batch_hs_before_layernorm, batch_labels):
        logits = vit_from_last_layer(hidden_states_before_layernorm=cache_state, tgt_pos=tgt_pos)
        # 出力されたlogitsを確率に変換  
        proba = torch.nn.functional.softmax(logits, dim=-1)
        pred_label = torch.argmax(proba, dim=-1)
        for pl, tl in zip(pred_label, y):
            all_pred_labels.append(pl.item())
            all_true_labels.append(tl)
    all_pred_labels = np.array(all_pred_labels)
    all_true_labels = np.array(all_true_labels)
    return all_pred_labels, all_true_labels

if __name__ == "__main__":
    # データセットをargparseで受け取る
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str)
    parser.add_argument('k', type=int, help="the fold id (0 to K-1)")
    parser.add_argument('tgt_rank', type=int, help="the rank of the target misclassification type")
    parser.add_argument("--setting_path", type=str, help="path to the setting json file", default=None)
    parser.add_argument("--fl_method", type=str, help="the method used for FL", default="vdiff")
    parser.add_argument('--misclf_type', type=str, help="the type of misclassification (src_tgt or tgt)", default="tgt")
    parser.add_argument('--only_eval', action='store_true', help="if True, only evaluate the saved patch", default=False)
    parser.add_argument("--custom_n", type=int, help="the custom n for the FL", default=None)
    parser.add_argument("--custom_alpha", type=float, help="the custom alpha for the repair", default=None)
    args = parser.parse_args()
    ds_name = args.ds
    k = args.k
    tgt_rank = args.tgt_rank
    setting_path = args.setting_path
    fl_method = args.fl_method
    misclf_type = args.misclf_type
    only_eval = args.only_eval
    custom_n = args.custom_n
    custom_alpha = args.custom_alpha
    print(f"ds_name: {ds_name}, fold_id: {k}, tgt_rank: {tgt_rank}, fl_method: {fl_method}, misclf_type: {misclf_type}")

    # TODO: あとでrandomly weights selectionも実装
    if fl_method == "random":
        NotImplementedError, "randomly weights selection is not implemented yet."
    # 設定のjsonファイルが指定された場合
    if setting_path is not None:
        assert os.path.exists(setting_path), f"{setting_path} does not exist."
        setting_dic = json2dict(setting_path)
        # setting_idは setting_{setting_id}.json というファイル名になる
        setting_id = os.path.basename(setting_path).split(".")[0].split("_")[-1]
    # 設定のjsonファイルが指定されない場合はnとalphaだけカスタムorデフォルトの設定を使う
    else:
        setting_dic = DEFAULT_SETTINGS
        setting_id = "default"
        if custom_n is not None:
            setting_dic["n"] = custom_n
            setting_id = f"n{custom_n}"
        if custom_alpha is not None:
            setting_dic["alpha"] = custom_alpha
            setting_id = f"alpha{custom_alpha}" if custom_n is None else f"n{custom_n}_alpha{custom_alpha}"
    # pretrained modelのディレクトリ
    pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR.format(k=k)
    # 結果とかログの保存先を先に作っておく
    save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_repair_weight_by_de")
    if fl_method == "vdiff":
        patch_save_path = os.path.join(save_dir, f"best_patch_{setting_id}.npy")
        tracker_save_path = os.path.join(save_dir, f"tracker_{setting_id}.pkl")
    elif fl_method == "random":
        patch_save_path = os.path.join(save_dir, f"best_patch_{setting_id}_random.npy")
        tracker_save_path = os.path.join(save_dir, f"tracker_{setting_id}_random.pkl")
    else:
        NotImplementedError
    os.makedirs(save_dir, exist_ok=True)
    # このpythonのファイル名を取得
    this_file_name = os.path.basename(__file__).split(".")[0]
    exp_name = f"{this_file_name}_{setting_id}"
    # loggerの設定をして設定情報を表示
    logger = set_exp_logging(exp_dir=save_dir, exp_name=exp_name)
    logger.info(f"ds_name: {ds_name}, fold_id: {k}, setting_path: {setting_path}")
    logger.info(f"setting_dic (id={setting_id}): {setting_dic}")

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
    # ラベルの取得 (shuffleされない)
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
    ori_tgt_ds = ds_preprocessed[tgt_split]
    ori_tgt_labels = labels[tgt_split]
    tgt_layer = 11 # NOTE: we only use the last layer for repairing
    logger.info(f"tgt_layer: {tgt_layer}, tgt_split: {tgt_split}")

    # ===============================================
    # localization phase
    # ===============================================

    location_save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_weights_location")
    location_save_path = os.path.join(location_save_dir, f"location_n{setting_dic['n']}_{fl_method}.npy")
    pos_before, pos_after = np.load(location_save_path, allow_pickle=True)
    # log表示
    logger.info(f"pos_before={pos_before}")
    logger.info(f"pos_after={pos_after}")
    logger.info(f"num(pos_to_fix)=num(pos_before)+num(pos_before)={len(pos_before)}+{len(pos_after)}={len(pos_before)+len(pos_after)}")

    # ===============================================
    # Data preparation for repair
    # ===============================================

    # tgt_rankの誤分類情報を取り出す
    misclf_info_dir = os.path.join(pretrained_dir, "misclf_info")
    misclf_pair, tgt_label, tgt_mis_indices = identfy_tgt_misclf(misclf_info_dir, tgt_split=tgt_split, tgt_rank=tgt_rank, misclf_type=misclf_type)
    indices_to_incorrect = tgt_mis_indices

    # original model の repair setの各サンプルに対する正解/不正解のインデックスを取得
    pred_res_dir = os.path.join(pretrained_dir, "pred_results", "PredictionOutput")
    ori_pred_labels, is_correct, indices_to_correct = get_ori_model_predictions(pred_res_dir, labels, tgt_split=tgt_split, misclf_type=misclf_type, tgt_label=tgt_label)
    logger.info(f"len(indices_to_correct): {len(indices_to_correct)}, len(indices_to_incorrect): {len(indices_to_incorrect)}")

    # 正解データからrepairに使う一定数だけランダムに取り出す
    sampled_indices_to_correct = sample_from_correct_samples(setting_dic["num_sampled_from_correct"], indices_to_correct)
    ori_sampled_indices_to_correct, ori_indices_to_incorrect = sampled_indices_to_correct.copy(), indices_to_incorrect.copy()
    # 抽出した正解データと，全不正解データを結合して1つのデータセットにする
    tgt_indices = sampled_indices_to_correct.tolist() + indices_to_incorrect.tolist() # .tolist() は 非破壊的method
    # tgt_indicesは全てユニークな値であることを保証
    assert len(tgt_indices) == len(set(tgt_indices)), f"len(tgt_indices): {len(tgt_indices)}, len(set(tgt_indices)): {len(set(tgt_indices))}"
    logger.info(f"tgt_indices: {tgt_indices} (len: {len(tgt_indices)})")
    # tgt_indicesに対応するデータトラベルを取り出す
    tgt_ds = ori_tgt_ds.select(tgt_indices)
    tgt_labels = ori_tgt_labels[tgt_indices]
    logger.info(f"ori_pred_labels[tgt_indices]: {ori_pred_labels[tgt_indices]} (len: {len(ori_pred_labels[tgt_indices])})")
    logger.info(f"ori_tgt_labels[tgt_indices]: {tgt_labels} (len: {len(tgt_labels)})")
    
    # repair setに対するhidden_states_before_layernormを取得
    hs_save_dir = os.path.join(pretrained_dir, f"cache_hidden_states_before_layernorm_{tgt_split}")
    hs_save_path = os.path.join(hs_save_dir, f"hidden_states_before_layernorm_{tgt_layer}.npy")
    assert os.path.exists(hs_save_path), f"{hs_save_path} does not exist."
    batch_hs_before_layernorm_tgt = get_batched_hs(hs_save_path, batch_size, tgt_indices, device=device)
    batch_labels_tgt = get_batched_labels(ori_tgt_labels, batch_size, tgt_indices)
    # repair set全体 (tgt_indicesを指定しない) のbatchも作成
    batch_hs_before_layernorm = get_batched_hs(hs_save_path, batch_size, device=device)
    batch_labels = get_batched_labels(ori_tgt_labels, batch_size)

    # repair に使ったデータの tgt_indices を npy で保存
    tgt_indices_save_path = os.path.join(save_dir, f"tgt_indices.npy") # TODO: tgt_indicesの特定にランダム性が入る場合はそれをトラックできるようなファイル名にする必要あり
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
    indices_to_incorrect_for_de_searcher = np.arange(len(sampled_indices_to_correct), len(indices_to_correct) + len(indices_to_incorrect))
    logger.info(f"indices_to_correct_for_de_searcher: {indices_to_correct_for_de_searcher}, indices_to_incorrect_for_de_searcher: {indices_to_incorrect_for_de_searcher}")
    # batch_hs_before_layernorm_tgtの最初のindices_to_correct_for_de_searcher個に対してはモデルが正解して，残りのindices_to_incorrect_for_de_searcher個に対してはモデルが不正解していることを確認
    assert np.sum(ori_is_correct_tgt[indices_to_correct_for_de_searcher]) == len(indices_to_correct_for_de_searcher), f"ori_is_correct_tgt[indices_to_correct_for_de_searcher]: {ori_is_correct_tgt[indices_to_correct_for_de_searcher]}"
    assert np.sum(ori_is_correct_tgt[indices_to_incorrect_for_de_searcher]) == 0, f"ori_is_correct_tgt[indices_to_incorrect_for_de_searcher]: {ori_is_correct_tgt[indices_to_incorrect_for_de_searcher]}"
    
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
        weight_med2after=weight_med2after
    )
    if not only_eval:
        logger.info(f"Start DE search...")
        s = time.perf_counter()
        patch, fitness_tracker = searcher.search(patch_save_path=patch_save_path, pos_before=pos_before, pos_after=pos_after, tracker_save_path=tracker_save_path)
        e = time.perf_counter()
        # print(f"[after DE] {vit_from_last_layer.base_model_last_layer.intermediate.dense.weight.data[pos_before[0][0]][pos_before[0][1]]}")
        tot_time = e - s
        logger.info(f"Total execution time: {tot_time} sec.")
        exit()
    else:
        logger.info(f"Only evaluation mode is on. Skippping the DE search.")
        tot_time = None
        # 保存したpatchを読み込む
        patch = np.load(patch_save_path)
        fitness_tracker = pickle.load(open(tracker_save_path, "rb"))

    # ===============================================
    # evaluation for the repair set
    # ===============================================
    assert len(patch) == len(pos_before) + len(pos_after), f"len(patch): {len(patch)}, len(pos_before)+len(pos_after): {len(pos_before)+len(pos_after)}"
    # patchを適切な位置の重みにセットする (ちゃんとできて
    # るはず)
    for ba, pos in enumerate([pos_before, pos_after]):
        # patch_candidateのindexを設定
        if ba == 0:
            idx_patch_candidate = range(0, len(pos_before))
            tgt_weight_data = vit_from_last_layer.base_model_last_layer.intermediate.dense.weight.data
        else:
            idx_patch_candidate = range(len(pos_before), len(pos_before) + len(pos_after))
            tgt_weight_data = vit_from_last_layer.base_model_last_layer.output.dense.weight.data
        # posで指定された位置のニューロンを書き換える
        xi, yi = pos[:, 0], pos[:, 1]
        tgt_weight_data[xi, yi] = torch.from_numpy(patch[idx_patch_candidate]).to(device)

    # 中間状態とラベルをバッチ化
    num_batches = (hs_before_layernorm.shape[0] + batch_size - 1) // batch_size  # バッチの数を計算 (最後の中途半端なバッチも使いたいので，切り上げ)
    batch_hs_before_layernorm = np.array_split(hs_before_layernorm, num_batches)
    batched_labels = make_batch_of_label(labels=ori_tgt_labels, num_batch=batch_size)

    all_proba = []
    for cached_state, y in zip(batch_hs_before_layernorm, batched_labels):
        logits = vit_from_last_layer(hidden_states_before_layernorm=cached_state, tgt_pos=tgt_pos)
        # 出力されたlogitsを確率に変換
        proba = torch.nn.functional.softmax(logits, dim=-1)
        all_proba.append(proba.detach().cpu().numpy())
    all_proba = np.concatenate(all_proba, axis=0) # (num_of_data, num_of_classes)
    new_pred_labels = np.argmax(all_proba, axis=-1) # (num_of_data, )

    # old_pred_labels と new_pred_labels それぞれの違いから以下の3つのメトリクスを算出
    # 1. Δaccuracy: acc after repair - acc before repair
    # 2. repair rate: 修正前モデルが誤ったデータのうち，修正後に正しく修正されたデータの割合
    # 3. break rete: 修正前モデルが正しかったデータのうち，修正後に誤って修正されたデータの割合
    # 上記のメトリクス算出後，setting_idに紐づいたファイルに保存

    # まずは修正履歴の正解/不正解の配列を作る
    is_correct_old = ori_pred_labels == ori_tgt_labels
    is_correct_new = new_pred_labels == ori_tgt_labels
    # 修正前後の正解率を計算
    acc_old = np.mean(is_correct_old)
    acc_new = np.mean(is_correct_new)
    # 修正前後の正解率の差を計算
    delta_acc = acc_new - acc_old
    # 修正率 (repair_rate) と破壊率 (break_rate) を計算
    repair_cnt = np.sum(~is_correct_old & is_correct_new) # 不正解 -> 正解のcnt
    break_cnt = np.sum(is_correct_old & ~is_correct_new) # 正解 -> 不正解のcnt
    repair_rate = repair_cnt / np.sum(~is_correct_old) # 不正解 -> 正解のcnt / 不正解のcnt
    break_rate = break_cnt / np.sum(is_correct_old) # 正解 -> 不正解のcnt / 正解のcnt
    logger.info(f"[Overall] repair_rate: {repair_rate} ({repair_cnt} / {np.sum(~is_correct_old)}), break_rate: {break_rate} ({break_cnt} / {np.sum(is_correct_old)})")
    # Below two are should be the same

    if misclf_type == "tgt":
        logger.info(f"new_pred_labels = \n{new_pred_labels[tgt_indices]}, len(new_pred_labels) = {len(new_pred_labels[tgt_indices])}")
        logger.info(f"fitness_tracker[all_pred_labels][-1] = \n{fitness_tracker['all_pred_labels'][-1]}, len(fitness_tracker[all_pred_labels][-1]) = {len(fitness_tracker['all_pred_labels'][-1])}")
        is_correct_new_from_fitness = fitness_tracker['all_pred_labels'][-1] == ori_tgt_labels[tgt_indices]
        # 以下が違うのはおかしい...
        print(is_correct_new[tgt_indices])
        print(np.sum(is_correct_new[tgt_indices]))
        print(is_correct_new_from_fitness)
        print(np.sum(is_correct_new_from_fitness))

        all_pred_labels = []
        all_true_labels = []
        for cache_state, y in zip(batch_hs_before_layernorm_tgt, batch_labels_tgt):
            logits = vit_from_last_layer(hidden_states_before_layernorm=cache_state, tgt_pos=tgt_pos)
            # 出力されたlogitsを確率に変換
            proba = torch.nn.functional.softmax(logits, dim=-1)
            pred_label = torch.argmax(proba, dim=-1)
            for pl, tl in zip(pred_label, y):
                all_pred_labels.append(pl.item())
                all_true_labels.append(tl)
        all_pred_labels = np.array(all_pred_labels)
        all_true_labels = np.array(all_true_labels)
        isc = all_pred_labels == all_true_labels
        print(f"sum(isc), len(isc), {sum(isc), len(isc)}")
        print(isc) # 最初の len(indices_to_correct) はTrue, それ以降はFalse
        print("all_pred_labels", all_pred_labels)
        print("all_true_labels", all_true_labels)

        # 元々の予測はori_indices_to_correctに対しては全部正解なことを保証
        assert sum(is_correct_old[ori_indices_to_correct]) == len(ori_indices_to_correct), f"sum(is_correct_old): {sum(is_correct_old)}, len(ori_indices_to_correct): {len(ori_indices_to_correct)}"
        # 元々の予測はtgt_mis_indicesに対しては全部不正解なことを保証
        assert sum(is_correct_old[ori_indices_to_incorrect]) == 0, f"sum(is_correct_new): {sum(is_correct_new)}"
        # repairに使ったデータの予測結果がどう変わったか？
        is_correct_new_tgt_to_correct = is_correct_new[ori_indices_to_correct]
        is_correct_new_tgt_to_incorrect = is_correct_new[ori_indices_to_incorrect]

        # 修正率 (repair_rate) と破壊率 (break_rate) を計算
        tgt_repair_cnt = np.sum(is_correct_new_tgt_to_incorrect)
        tgt_break_cnt = np.sum(~is_correct_new_tgt_to_correct)
        tgt_repair_rate = tgt_repair_cnt / np.sum(~is_correct_old[ori_indices_to_incorrect])
        tgt_break_rate = tgt_break_cnt / np.sum(is_correct_old[ori_indices_to_correct])

        logger.info(f"[Target] repair_rate: {tgt_repair_rate} ({tgt_repair_cnt} / {np.sum(~is_correct_old[ori_indices_to_incorrect])}), break_rate: {tgt_break_rate} ({tgt_break_cnt} / {np.sum(is_correct_old[ori_indices_to_correct])})")
        searcher.eval_weights(patch, pos_before, pos_after) # 確認用
        exit()
        
        # tgt_labelへのf1を計算
        f1_score =  evaluate.load("f1")
        tgt_f1_old = f1_score.compute(predictions=ori_pred_labels, references=ori_tgt_labels, average=None)["f1"][tgt_label]
        tgt_f1_new = f1_score.compute(predictions=new_pred_labels, references=ori_tgt_labels, average=None)["f1"][tgt_label]
        delta_f1 = tgt_f1_new - tgt_f1_old

    # メトリクスを保存
    metrics = {
        "delta_acc": delta_acc,
        "repair_rate": repair_rate,
        "break_rate": break_rate,
        "tot_time": tot_time,
        "acc_old": acc_old,
        "acc_new": acc_new,
        "tgt_rep_rate": tgt_repair_rate,
        "tgt_brk_rate": tgt_break_rate,
        "delta_f1": delta_f1,
        "tgt_f1_old": tgt_f1_old,
        "tgt_f1_new": tgt_f1_new
    }
    logger.info(metrics)
    print(metrics)
    if fl_method == "vdiff":
        metrics_dir = os.path.join(save_dir, f"metrics_for_repair_{setting_id}.json")
    elif fl_method == "random":
        metrics_dir = os.path.join(save_dir, f"metrics_for_repair_{setting_id}_random.json")
    else:
        NotImplementedError
    # only_eval時は保存しない (一度only_evalなしで実行して保存されている前提なので)
    if not only_eval:
        with open(metrics_dir, "w") as f:
            json.dump(metrics, f)