import os, sys, time, json
from tqdm import tqdm
import numpy as np
import torch
import argparse
from datasets import load_from_disk
from transformers import ViTForImageClassification
from utils.helper import get_device, get_corruption_types
from utils.vit_util import transforms, transforms_c100, get_vscore
from utils.constant import ViTExperiment

if __name__ == "__main__":
    # プログラム引数の受け取り
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str)
    parser.add_argument("tgt_ct", type=str)
    parser.add_argument('--used_column', type=str, default="train")
    parser.add_argument('--start_layer_idx', type=int, default=9)
    parser.add_argument('--severity', type=int, help="severity of corruption (integer from 0 to 4). when set to -1, treat all as one dataset.", default=4)
    parser.add_argument('--include_ori', action="store_true", default=False)
    args = parser.parse_args()
    ori_ds_name = args.ds
    tgt_ct = args.tgt_ct
    start_layer_idx = args.start_layer_idx
    used_column = args.used_column
    severity = args.severity
    include_ori = args.include_ori
    # argparseで受け取った引数のサマリーを表示
    print(f"ori_ds_name: {ori_ds_name}, tgt_ct: {tgt_ct}, start_layer_idx: {start_layer_idx}, used_column: {used_column}, severity: {severity}, include_ori: {include_ori}")

    # datasetごとに違う変数のセット
    if ori_ds_name == "c10":
        tf_func = transforms
        label_col = "label"
        num_labels = 10
    elif ori_ds_name == "c100":
        tf_func = transforms_c100
        label_col = "fine_label"
        num_labels = 100
    else:
        NotImplementedError

    # デバイス (cuda, or cpu) の取得
    device = get_device()
    # get corruption types
    ct_list = get_corruption_types()
    # original datasetをロード
    ori_ds = load_from_disk(os.path.join(ViTExperiment.DATASET_DIR, ori_ds_name))[used_column]
    ori_labels = np.array(ori_ds[label_col])
    ori_ds = ori_ds.with_transform(tf_func)
    # tgt_ctに対するcorruption datasetをロード
    ds = load_from_disk(os.path.join(ViTExperiment.DATASET_DIR, f"{ori_ds_name}c_severity{severity}", tgt_ct))
    # train, testに分ける
    ds_split = ds.train_test_split(test_size=0.4, shuffle=True, seed=777)[used_column] # XXX: !SEEDは絶対固定!
    labels = np.array(ds_split[label_col])
    ct_ds = ds_split.with_transform(tf_func)
    # dsとct_dsのデータセットのサイズを出力
    print(f"ori_ds: {len(ori_ds)}, ct_ds: {len(ct_ds)}")
    # pretrained modelのロード
    pretrained_dir = getattr(ViTExperiment, ori_ds_name).OUTPUT_DIR
    result_dir = os.path.join(getattr(ViTExperiment, ori_ds_name).OUTPUT_DIR, "neuron_scores")
    model = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
    model.eval()
    batch_size = ViTExperiment.BATCH_SIZE
    tgt_pos = ViTExperiment.CLS_IDX
    # 対象レイヤの設定
    start_li = start_layer_idx
    end_li = model.vit.config.num_hidden_layers

    tic = time.perf_counter()
    for ds_name, ds, ls in zip(["ori_ds", "ct_ds"], [ori_ds, ct_ds], [ori_labels, labels]):
        if not include_ori and ds_name == "ori_ds":
            continue
        # tgt_dsの設定
        print(f"{'='*60}\ntgt_ds: {ds_name}\n{'='*60}")
        tgt_ds = ds
        print(f"len(tgt_ds): {len(tgt_ds)}")
        
        # loop for dataset batch
        all_mid_states = []
        all_logits = []
        for entry_dic in tqdm(tgt_ds.iter(batch_size=batch_size), total=len(tgt_ds)//batch_size+1):
            x, _ = entry_dic["pixel_values"].to(device), entry_dic["labels"]
            output = model.forward(x, output_intermediate_states=True) # (batch_size, num_layers, num_neurons)
            # CLSトークンに対応するintermediate statesを取得
            output_mid_states = np.array([mid_states_each_layer[:, tgt_pos, :].cpu().detach().numpy()
                                                for mid_states_each_layer in output.intermediate_states])
            output_mid_states = output_mid_states.transpose(1, 0, 2) # (batch_size, num_layers, num_neurons)
            logits = output.logits.cpu().detach().numpy()
            all_logits.append(logits)
            all_mid_states.append(output_mid_states)
        all_logits = np.concatenate(all_logits) # (num_samples, num_labels)
        all_pred_labels = np.argmax(all_logits, axis=-1) # (num_samples, )
        all_mid_states = np.concatenate(all_mid_states) # (num_samples, num_layers, num_neurons)
        # サンプルごとの予測の正解不正解の配列を作る
        is_correct = all_pred_labels == ls
        # あっていたか否かでmid_statesを分ける
        correct_mid_states = all_mid_states[is_correct]
        incorrect_mid_states = all_mid_states[~is_correct]
        print(f"len(correct_mid_states), len(incorrect_mid_states) = {len(correct_mid_states), len(incorrect_mid_states)}")
        for cor_mis, mid_states in zip(["cor", "mis"], [correct_mid_states, incorrect_mid_states]):
            print(f"cor_mis: {cor_mis}, len({cor_mis}_states): {len(mid_states)}")
            # 対象のレイヤに対してvscoreを計算
            # =================================
            vscore_per_layer = []
            for tgt_layer in range(start_li, end_li):
                print(f"tgt_layer: {tgt_layer}")
                tgt_mid_state = mid_states[:, tgt_layer, :] # (num_samples, num_neurons)
                # vscoreを計算
                vscore = get_vscore(tgt_mid_state)
                vscore_per_layer.append(vscore)
            vscores = np.array(vscore_per_layer) # (num_tgt_layer, num_neurons)
            # vscoresを保存
            ds_type = f"ori_{used_column}" if ds_name == "ori_ds" else f"{tgt_ct}_{used_column}"
            vscore_save_path = os.path.join(result_dir, f"vscore_l{start_li}tol{end_li}_all_label_{ds_type}_{cor_mis}.npy")
            np.save(vscore_save_path, vscores)
            print(f"vscore ({vscores.shape}) saved at {vscore_save_path}") # mid_statesがnan (correct or incorrect predictions の数が 0) の場合はvscoreもnanになる
            
            # vscoreの高いニューロン位置を保存
            # =================================
            threshold = np.percentile(vscores.flatten(), 97) # NOTE: 3%の閾値 (hard-coding)
            # 閾値以上の値を持つ要素のインデックスを取得
            indices = np.argwhere(vscores >= threshold)
            # 上位3%のインデックス (レイヤ番号, ニューロン番号) のリスト
            vn = [(int(l_id+start_li), int(n_id)) for l_id, n_id in indices]# 結果を保存
            print(vn)
            # 結果を保存
            save_dict = {}
            save_dict["num_kn"] = len(vn)
            save_dict["num_kn_per_layer"] = {l: len([n_id for l_id, n_id in vn if l_id == l]) for l in range(start_li, end_li)}
            save_dict["kn"] = vn
            json_save_path = vscore_save_path.replace("npy", "json")
            with open(json_save_path, "w") as f:
                json.dump(save_dict, f, indent=4)
            print(f"vn information (json) is saved at {json_save_path}")

    toc = time.perf_counter()
    print(f"***** Costing time: {toc - tic:0.4f} seconds *****")
