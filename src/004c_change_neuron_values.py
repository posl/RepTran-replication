import os, sys, time, pickle, json
import argparse
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import torch
from datasets import load_from_disk
from transformers import ViTForImageClassification
from utils.helper import get_device, json2dict
from utils.vit_util import transforms, transforms_c100, ViTFromLastLayer, localize_neurons, localize_neurons_random
from utils.data_util import make_batch_of_label
from utils.constant import ViTExperiment
from utils.log import set_exp_logging
from utils.de import DE_searcher
from logging import getLogger

logger = getLogger("base_logger")

DEFAULT_SETTINGS = {
    "theta": 10, 
    "num_sampled_from_correct": 200,
    "max_search_num": 50,
    "pop_size": 100,
    "alpha": 1
}

if __name__ == "__main__":
    # データセットをargparseで受け取る
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str)
    parser.add_argument('k', type=int, help="the fold id (0 to K-1)")
    parser.add_argument("--setting_path", type=str, help="path to the setting json file", default=None)
    parser.add_argument("--fl_method", type=str, help="the method used for FL", default="vdiff")
    args = parser.parse_args()
    ds_name = args.ds
    k = args.k
    fl_method = args.fl_method
    setting_path = args.setting_path
    # 設定のjsonファイルが指定された場合
    if setting_path is not None:
        assert os.path.exists(setting_path), f"{setting_path} does not exist."
        setting_dic = json2dict(setting_path)
        # setting_idは setting_{setting_id}.json というファイル名になる
        setting_id = os.path.basename(setting_path).split(".")[0].split("_")[-1]
    # 設定のjsonファイルが指定されない場合はデフォルトの設定を使う
    else:
        setting_dic = DEFAULT_SETTINGS
        setting_id = "default"
    # pretrained modelのディレクトリ
    pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR.format(k=k)
    # このpythonのファイル名を取得
    this_file_name = os.path.basename(__file__).split(".")[0]
    # loggerの設定をして設定情報を表示
    logger = set_exp_logging(exp_dir=pretrained_dir, exp_name=this_file_name)
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
    ori_tgt_ds = ds_preprocessed[tgt_split]
    ori_tgt_labels = labels[tgt_split]
    tgt_layer = 11 # NOTE: we only use the last layer for repairing
    print(f"tgt_layer: {tgt_layer}, tgt_split: {tgt_split}")
    logger.info(f"tgt_layer: {tgt_layer}, tgt_split: {tgt_split}")

    # ===============================================
    # localization phase
    # ===============================================

    if fl_method == "vdiff":
        localizer = localize_neurons
    elif fl_method == "random":
        localizer = localize_neurons_random

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
    places_to_fix, tgt_vdiff = localizer(vmap_dic, tgt_layer, theta=setting_dic["theta"])
    logger.info(f"places_to_fix={places_to_fix}")
    logger.info(f"num(location)={len(places_to_fix)}")

    # ===============================================
    # patch generation phase
    # ===============================================

    # original model の repair setの各サンプルに対する正解/不正解のインデックスを取得
    pred_res_dir = os.path.join(pretrained_dir, "pred_results", "PredictionOutput")
    with open(os.path.join(pred_res_dir, f"{tgt_split}_pred.pkl"), "rb") as f:
        pred_res = pickle.load(f)
    pred_logits = pred_res.predictions
    ori_pred_labels = np.argmax(pred_logits, axis=-1)
    is_correct = ori_pred_labels == labels[tgt_split]
    indices_to_correct = np.where(is_correct)[0]
    indices_to_incorrect = np.where(~is_correct)[0]
    logger.info(f"len(indices_to_correct): {len(indices_to_correct)}, len(indices_to_incorrect): {len(indices_to_incorrect)}")

    # 正解データからrepairに使う一定数だけランダムに取り出す
    num_sampled_from_correct = setting_dic["num_sampled_from_correct"]
    indices_to_correct = np.random.choice(indices_to_correct, num_sampled_from_correct, replace=False)
    # 抽出した正解データと，全不正解データを結合して1つのデータセットにする
    tgt_ds = ori_tgt_ds.select(indices_to_correct.tolist() + indices_to_incorrect.tolist())
    tgt_labels = ori_tgt_labels[indices_to_correct.tolist() + indices_to_incorrect.tolist()]
    
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
    max_search_num = setting_dic["max_search_num"]
    alpha = setting_dic["alpha"]
    patch_aggr = alpha * len(indices_to_correct) / len(indices_to_incorrect)
    logger.info(f"alpha of the fitness func.: {patch_aggr}")
    pop_size = setting_dic["pop_size"]
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
        pop_size=pop_size,
    )

    save_dir = os.path.join(pretrained_dir, "repairing_by_de")
    if fl_method == "vdiff":
        save_path = os.path.join(save_dir, f"best_patch_{setting_id}.npy")
    elif fl_method == "random":
        save_path = os.path.join(save_dir, f"best_patch_{setting_id}_random.npy")
    else:
        NotImplementedError
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Start DE search...")
    s = time.perf_counter()
    searcher.search(places_to_fix, save_path=save_path)
    e = time.perf_counter()
    tot_time = e - s
    logger.info(f"Total execution time: {tot_time} sec.")

    # ===============================================
    # evaluation for the repair set
    # ===============================================
    # 保存したpatchを読み込む
    patch = np.load(save_path)
    # repair setの全サンプルに対する中間状態とラベルを再ロード
    hs_before_layernorm = torch.from_numpy(np.load(hs_save_path)).to(device)
    # 中間状態とラベルをバッチ化
    num_batches = (hs_before_layernorm.shape[0] + batch_size - 1) // batch_size  # バッチの数を計算 (最後の中途半端なバッチも使いたいので，切り上げ)
    batch_hs_before_layernorm = np.array_split(hs_before_layernorm, num_batches)
    batched_labels = make_batch_of_label(labels=ori_tgt_labels, num_batch=batch_size)

    all_proba = []
    for cached_state, y in zip(batch_hs_before_layernorm, batched_labels):
        logits = vit_from_last_layer(hidden_states_before_layernorm=cached_state, tgt_pos=tgt_pos, imp_pos=places_to_fix, imp_op=patch)
        # 出力されたlogitsを確率に変換
        proba = torch.nn.functional.softmax(logits, dim=-1)
        all_proba.append(proba.detach().cpu().numpy())
    all_proba = np.concatenate(all_proba, axis=0) # (num_of_data, num_of_classes)
    new_pred_laebls = np.argmax(all_proba, axis=-1) # (num_of_data, )

    # old_pred_labels と new_pred_labels それぞれの違いから以下の3つのメトリクスを算出
    # 1. Δaccuracy: acc after repair - acc before repair
    # 2. repair rate: 修正前モデルが誤ったデータのうち，修正後に正しく修正されたデータの割合
    # 3. break rete: 修正前モデルが正しかったデータのうち，修正後に誤って修正されたデータの割合
    # 上記のメトリクス算出後，setting_idに紐づいたファイルに保存

    # まずは修正履歴の正解/不正解の配列を作る
    is_correct_old = ori_pred_labels == ori_tgt_labels
    is_correct_new = new_pred_laebls == ori_tgt_labels
    # 修正前後の正解率を計算
    acc_old = np.mean(is_correct_old)
    acc_new = np.mean(is_correct_new)
    # 修正前後の正解率の差を計算
    delta_acc = acc_new - acc_old
    # 修正前後の正解率の差を計算
    repair_rate = np.mean(~is_correct_old & is_correct_new)
    break_rate = np.mean(is_correct_old & ~is_correct_new)

    # メトリクスを保存
    metrics = {
        "delta_acc": delta_acc,
        "repair_rate": repair_rate,
        "break_rate": break_rate,
        "tot_time": tot_time,
        "acc_old": acc_old,
        "acc_new": acc_new
    }
    logger.info(f"delta_acc: {delta_acc}, repair_rate: {repair_rate}, break_rate: {break_rate}")
    if fl_method == "vdiff":
        metrics_dir = os.path.join(save_dir, f"metrics_for_repair_{setting_id}.json")
    elif fl_method == "random":
        metrics_dir = os.path.join(save_dir, f"metrics_for_repair_{setting_id}_random.json")
    else:
        NotImplementedError
    with open(metrics_dir, "w") as f:
        json.dump(metrics, f)