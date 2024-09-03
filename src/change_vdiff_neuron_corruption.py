import os, sys, time, math
from tqdm import tqdm
import json
import numpy as np
import pandas as pd
import torch
import argparse
import evaluate
from datasets import load_from_disk
from transformers import ViTForImageClassification, DefaultDataCollator, Trainer
from utils.helper import get_device, get_corruption_types
from utils.vit_util import transforms, transforms_c100, compute_metrics, processor, pred_to_labels
from utils.constant import ViTExperiment

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
    # プログラム引数の受け取り
    parser = argparse.ArgumentParser(description='start_layer_idx selector')
    parser.add_argument('ds_name', type=str)
    parser.add_argument('tgt_ct', type=str)
    parser.add_argument('--severity', type=int, help="severity of corruption (integer from 0 to 4). when set to -1, treat all as one dataset.", default=4)
    parser.add_argument('--used_column', type=str, default="train")
    parser.add_argument('--start_layer_idx', type=int, default=0)
    args = parser.parse_args()
    ds_name = args.ds_name
    tgt_ct = args.tgt_ct
    severity = args.severity
    used_column = args.used_column
    start_layer_idx = args.start_layer_idx
    batch_size = 32
    tgt_pos = ViTExperiment.CLS_IDX # tgt_posはCLS_IDXで固定 (intermediate_statesの2次元目の0番目の要素に対応する中間層ニューロン)
    print(f"ds_name: {ds_name}, tgt_ct: {tgt_ct}, severity: {severity}, used_column: {used_column}")

    # datasetごとに違う変数のセット
    if ds_name == "c10c":
        tf_func = transforms
        label_col = "label"
        num_labels = 10
        ori_ds_name = "c10"
    elif ds_name == "c100c":
        tf_func = transforms_c100
        label_col = "fine_label"
        num_labels = 100
        ori_ds_name = "c100"
    else:
        NotImplementedError
    
    # デバイス (cuda, or cpu) の取得
    device = get_device()
    ct_list = get_corruption_types()
    dataset_dir = ViTExperiment.DATASET_DIR
    data_collator = DefaultDataCollator()
    pretrained_dir = getattr(ViTExperiment, ori_ds_name).OUTPUT_DIR 

    # tgt_ctがct_list+["ori"]の中になかったらエラー
    if tgt_ct not in ct_list + ["ori"]:
        raise ValueError(f"tgt_ct must be in {ct_list + ['ori']}")
    
    # オリジナルのモデル (ノーマルのC100でftしたモデル) をロード
    model = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
    training_args = torch.load(os.path.join(pretrained_dir, "training_args.bin"))
    result_dir = os.path.join(getattr(ViTExperiment, ori_ds_name).OUTPUT_DIR, f"{tgt_ct}_severity{severity}", "vmap")
    # 対象レイヤの設定
    start_li = start_layer_idx
    end_li = model.vit.config.num_hidden_layers
    tgt_layer = 11 # vmap.ipynbの調査結果から，最終層だけ対象にする

    # vmap_diffのtop10%のニューロンの位置を取得
    vmap_diff_all = np.load(os.path.join(result_dir, f"vmap_diff_{used_column}_l{start_li}tol{end_li}.npy")) # (num_of_neurons, num_of_layers)
    vmap_diff = vmap_diff_all[:, tgt_layer] # (num_of_neurons,)
    top10_neuron_loc = np.load(os.path.join(result_dir, f"top10_idx_{used_column}_l{tgt_layer}.npy")) # (num_of_neurons*0.1, )
    print(f"vmap_diff: {vmap_diff.shape}, top10_neuron_loc: {top10_neuron_loc.shape}")
    # top10%のニューロンのに対応するvmap_diffの値を取得
    top10_vmap_diff = vmap_diff[top10_neuron_loc]
    # ニューロンの位置とvdiffの値をdfにする
    top10_df = pd.DataFrame({"neuron_loc": top10_neuron_loc, "vmap_diff": top10_vmap_diff})
    # vdiffの値がプラスの時は2，マイナスの時は0になるような列を作る
    top10_df["operation"] = top10_df["vmap_diff"].apply(lambda x: 2 if x > 0 else 0)

    # 予測結果のメトリクスを格納するdataframe
    header = "operation," + ",".join(["ori"] + ct_list) # 列名
    # headerを列として持つdataframeを作成
    acc_df = pd.DataFrame(columns=header.split(","))
    f1_df = pd.DataFrame(columns=header.split(","))
    # 結果を保存するdirectory
    save_dir = os.path.join(getattr(ViTExperiment, ori_ds_name).OUTPUT_DIR, "pred_results_all_label_by_vdiff")
    os.makedirs(save_dir, exist_ok=True)

    tic = time.perf_counter()
    # 各ctのtest setに対する予測を行う
    for i, ct in enumerate(["ori"] + ct_list):
        print(f"{'='*60}\npred_tgt_ds: {ct} ({i+1}/{len(['ori'] + ct_list)})\n")
        if ct == "ori":
            ds = load_from_disk(os.path.join(dataset_dir, ori_ds_name))["test"]
            labels = np.array(ds[label_col])
            tgt_ds = ds.with_transform(tf_func)
        else:
            # 対象のcorruption, severityのdatasetをロード
            ds = load_from_disk(os.path.join(dataset_dir, f"{ds_name}_severity{severity}", ct))
            # datasetをtrain, testに分ける
            ds_split = ds.train_test_split(test_size=0.4, shuffle=True, seed=777)["test"] # XXX: !SEEDは絶対固定!
            labels = np.array(ds_split[label_col])
            # 読み込まれた時にリアルタイムで前処理を適用するようにする
            tgt_ds = ds_split.with_transform(tf_func)

        # enhance: vdiff (vcor-vmis) が大きいニューロンを増幅, suppress: vdiffが小さいニューロンを抑制, both: 両方
        for op in ["enhance", "suppress", "both"]:
            print(f"{'='*60}\nop: {op}\n{'='*60}")
            print("current acc_df")
            print(acc_df)
                
            # dfの前処理
            # "dataset" 列 が f"{tgt_ct}_{cor_mis}" であり, かつ, "operation" 列が op である行があるかどうかをチェック
            condition = (acc_df['operation'] == {op})
            # 条件に合う行が存在しない場合
            if acc_df[condition].empty:
                # acc, f1_dfの新しい行を作成
                new_row_dict = {"operation": op}
                # new_row_dictをacc_dfとf1_dfに追加
                acc_df = acc_df.append(new_row_dict, ignore_index=True)
                f1_df = f1_df.append(new_row_dict, ignore_index=True)
                # conditionを更新 (上のconditionから行が変更されたので再評価必要)
                condition = acc_df['operation'] == op
            # この時点でconditionに合う行は1つでないといけない
            assert len(acc_df[condition]) == 1

            if op == "enhance":
                # enhance: vdiff (vcor-vmis) が大きいニューロンを増幅
                is_enhanced = top10_df["vmap_diff"] > 0
                vn_pos = top10_df[is_enhanced]["neuron_loc"].values
                delta = top10_df[is_enhanced]["operation"].values
            elif op == "suppress":
                # suppress: vdiffが小さいニューロンを抑制
                is_suppressed = top10_df["vmap_diff"] <= 0
                vn_pos = top10_df[is_suppressed]["neuron_loc"].values
                delta = top10_df[is_suppressed]["operation"].values
            elif op == "both":
                # both: 両方
                vn_pos = top10_df["neuron_loc"].values
                delta = top10_df["operation"].values
            else:
                raise ValueError(f"op must be in ['enhance', 'suppress', 'both']")
            
            # 推論
            all_proba = []
            print(f"vn_pos: {len(vn_pos)}, delta: {len(delta)}")
            # loop for the dataset
            for data_idx, entry_dic in tqdm(enumerate(tgt_ds.iter(batch_size=batch_size)), 
                                    total=math.ceil(len(tgt_ds)/batch_size)):
                x, y = entry_dic["pixel_values"].to(device), entry_dic["labels"][0]
                # imp_posはレイヤ番号とニューロン番号のリスト
                imp_pos = [[tgt_layer, pos] for pos in vn_pos]
                assert len(imp_pos) == len(delta)
                # バッチに対応するhidden statesとintermediate statesの取得
                outputs = model(x, tgt_pos=tgt_pos, tgt_layer=start_li, imp_pos=imp_pos, imp_op=delta)
                # outputs.logitsを確率にする
                proba = torch.nn.functional.softmax(outputs.logits, dim=-1)
                all_proba.append(proba.detach().cpu().numpy())
            all_proba = np.concatenate(all_proba, axis=0) # (num_of_data, num_of_classes)
            # all_probaをnpyで保存
            save_path = os.path.join(save_dir, f"test_proba_vscore_l{start_li}tol{end_li}_{tgt_ct}_to_{ct}_{op}.npy")
            np.save(save_path, all_proba)
            print(f"proba: {all_proba.shape} is saved at {save_path}")

            # 予測ラベルを取得
            pred_labels = pred_to_labels(all_proba)
            # acc, f1を計算
            met_acc = evaluate.load("accuracy")
            met_f1 = evaluate.load("f1")
            acc = met_acc.compute(references=labels, predictions=pred_labels)["accuracy"]
            f1 = met_f1.compute(references=labels, predictions=pred_labels, average="macro")["f1"]
            print(f"acc: {acc}, f1: {f1}")
            
            # ここまでで acc_df, f1_df それぞれの1セルがえられる
            # conditionの行のctの列に対してデータを格納
            acc_df.loc[condition, ct] = acc
            f1_df.loc[condition, ct] = f1
                
    # acc_df, f1_df をそれぞれcsvで保存
    acc_save_path = os.path.join(save_dir, f"change_vn_acc_{tgt_ct}.csv")
    f1_save_path = os.path.join(save_dir, f"change_vn_f1_{tgt_ct}.csv")
    acc_df.to_csv(acc_save_path, index=False)
    f1_df.to_csv(f1_save_path, index=False)
    
    toc = time.perf_counter()
    print(f"Total time: {toc - tic} sec")