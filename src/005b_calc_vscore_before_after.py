import os, sys, time
import argparse
from tqdm import tqdm
import numpy as np
import torch
from datasets import load_from_disk
from transformers import ViTForImageClassification
from utils.helper import get_device
from utils.vit_util import transforms, transforms_c100, get_vscore
from utils.constant import ViTExperiment

if __name__ == "__main__":
    # データセットをargparseで受け取る
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str)
    parser.add_argument('k', type=int, help="the fold id (0 to K-1)")
    args = parser.parse_args()
    ds_name = args.ds
    k = args.k
    print(f"ds_name: {ds_name}, fold_id: {k}")

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
    split_names = list(ds.keys()) # [train, repair, test]
    # target_split_names = ["train", "repair"]
    target_split_names = ["repair"]
    # target_split_namesは全てsplit_namesに含まれていることを前提とする
    assert all([split_name in split_names for split_name in target_split_names]), f"target_split_names should be subset of split_names"
    # ラベルの取得
    labels = {
        "train": np.array(ds["train"][label_col]),
        "repair": np.array(ds["repair"][label_col]),
        "test": np.array(ds["test"][label_col])
    }
    # 読み込まれた時にリアルタイムで前処理を適用するようにする
    ds_preprocessed = ds.with_transform(tf_func)
    # pretrained modelのロード
    pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR.format(k=k)
    model = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
    model.eval()
    end_li = model.vit.config.num_hidden_layers
    batch_size = ViTExperiment.BATCH_SIZE

    for split in target_split_names:
        print(f"Process {split}...")
        # 必要なディレクトリがない場合は先に作っておく
        vscore_before_dir = os.path.join(getattr(ViTExperiment, ds_name).OUTPUT_DIR.format(k=k), "vscores_before")
        vscore_after_dir = os.path.join(getattr(ViTExperiment, ds_name).OUTPUT_DIR.format(k=k), "vscores_after")
        os.makedirs(vscore_before_dir, exist_ok=True)
        os.makedirs(vscore_after_dir, exist_ok=True)
        all_bhs = []
        all_ahs = []
        all_logits = []
        # loop for dataset batch
        for entry_dic in tqdm(ds_preprocessed[split].iter(batch_size=batch_size), total=len(ds_preprocessed[split])//batch_size+1):
            x, y = entry_dic["pixel_values"].to(device), entry_dic["labels"]
            output = model.forward(x, tgt_pos=tgt_pos, output_hidden_states_before_layernorm=False, output_intermediate_states=True)
            # CLSトークンに対応するintermediate statesを取得
            num_layer = len(output.intermediate_states)
            bhs, ahs = [], []
            for i in range(num_layer):
                # output.intermediate_states[i]はiレイヤめの(中間ニューロンの前のニューロン, 中間ニューロン，中間ニューロンの後のニューロン)がタプルで入っている
                bhs.append(output.intermediate_states[i][0])
                ahs.append(output.intermediate_states[i][2]) # ここの時点ではレイヤの次元が先に来る
            bhs = np.array(bhs).transpose(1, 0, 2) # (batch_size, num_layers, num_neurons)
            ahs = np.array(ahs).transpose(1, 0, 2) # (batch_size, num_layers, num_neurons)
            all_bhs.append(bhs)
            all_ahs.append(ahs)
            logits = output.logits.cpu().detach().numpy()
            all_logits.append(logits)
        all_logits = np.concatenate(all_logits) # (num_samples, num_labels)
        all_pred_labels = np.argmax(all_logits, axis=-1) # (num_samples, )
        all_bhs = np.concatenate(all_bhs)
        all_ahs = np.concatenate(all_ahs)
        # サンプルごとの予測の正解不正解の配列を作る
        is_correct = all_pred_labels == labels[split]
        # あっていたか否かでmid_statesを分ける
        correct_bhs = all_bhs[is_correct]
        incorrect_bhs = all_bhs[~is_correct]
        correct_ahs = all_ahs[is_correct]
        incorrect_ahs = all_ahs[~is_correct]
        print(f"len(correct_bhs), len(incorrect_bhs) = {len(correct_bhs), len(incorrect_bhs)}")
        print(f"len(correct_ahs), len(incorrect_ahs) = {len(correct_ahs), len(incorrect_ahs)}")
        assert len(correct_bhs) == len(correct_ahs), f"len(correct_bhs) != len(correct_ahs)"
        assert len(incorrect_bhs) == len(incorrect_ahs), f"len(incorrect_bhs) != len(incorrect_ahs)"

        # 正解/不正解データに対するv-scoreの計算を行い，保存する
        for correct_mid_states, incorrect_mid_states, vscore_dir in zip([correct_bhs, correct_ahs], [incorrect_bhs, incorrect_ahs], [vscore_before_dir, vscore_after_dir]): # bhs, ahs での繰り返し
            for cor_mis, mid_states in zip(["cor", "mis"], [correct_mid_states, incorrect_mid_states]):
                print(f"cor_mis: {cor_mis}, len({cor_mis}_states): {len(mid_states)}") # correct, incorrectでの繰り返し
                # 対象のレイヤに対してvscoreを計算
                # =================================
                vscore_per_layer = []
                for tgt_layer in range(end_li):
                    print(f"tgt_layer: {tgt_layer}")
                    tgt_mid_state = mid_states[:, tgt_layer, :] # (num_samples, num_neurons)
                    # vscoreを計算
                    vscore = get_vscore(tgt_mid_state)
                    vscore_per_layer.append(vscore)
                vscores = np.array(vscore_per_layer) # (num_tgt_layer, num_neurons)
                # vscoresを保存
                ds_type = f"ori_{split}"
                vscore_save_path = os.path.join(vscore_dir, f"vscore_l1tol{end_li}_all_label_{ds_type}_{cor_mis}.npy")
                np.save(vscore_save_path, vscores)
                print(f"vscore ({vscores.shape}) saved at {vscore_save_path}") # mid_statesがnan (correct or incorrect predictions の数が 0) の場合はvscoreもnanになる
        
        # NOTE: 中間ニューロン状態自体の保存は必要そうになったら実装する
        # # 各サンプルに対するレイヤごとの隠れ状態を保存していく
        # # 将来的なことを考えてnumpy->tensorに変換してから保存
        # num_layers = model.vit.config.num_hidden_layers
        # for l_idx in range(num_layers):
        #     # 特定のレイヤのstatesだけ抜き出し
        #     tgt_mid_states = torch.tensor(all_mid_states[:, l_idx, :]).cpu()
        #     # 保存
        #     intermediate_save_path = os.path.join(cache_dir, f"intermediate_states_l{l_idx}.pt")
        #     torch.save(tgt_mid_states, intermediate_save_path)
        #     print(f"tgt_mid_states: {tgt_mid_states.shape} is saved at {intermediate_save_path}")