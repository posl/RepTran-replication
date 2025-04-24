from utils.vit_util import transforms_c100, get_batched_hs, get_batched_labels, ViTFromLastLayer, get_ori_model_predictions, identfy_tgt_misclf
from utils.constant import ViTExperiment, Experiment3, ExperimentRepair1, Experiment4
from utils.helper import get_device
from utils.de import set_new_weights
from transformers import ViTForImageClassification
from tqdm import tqdm
import torch
import torch.nn as nn
import os
from collections import defaultdict
import numpy as np
import pandas as pd
from datasets import load_from_disk

# デバイスの設定
device = get_device()

def get_location_path(n, w_num, beta, fl_method, location_dir, generate_random=True):
    if fl_method == "ours":
        location_file = f"exp-fl-7_location_n{n}_beta{beta}_weight_ours.npy"
    elif fl_method == "bl":
        location_file = f"exp-fl-2_location_n{n}_weight_bl.npy"
    elif fl_method == "random":
        location_file = f"exp-fl-1_location_n{n}_weight_random.npy"
    else:
        raise ValueError(f"Unknown fl_method: {fl_method}")
    location_path = os.path.join(location_dir, location_file)
    return location_path

def get_loss_diff_path(n, w_num, beta, fl_method, loss_diff_dir, op, cor_mis):
    assert cor_mis in ["cor", "mis"], f"Unknown cor_mis: {cor_mis}"
    if fl_method == "ours":
        loss_diff_file = f"exp-fl-7_loss_diff_n{n}_beta{beta}_{op}_{cor_mis}_weight_ours.npy"
    elif fl_method == "bl":
        loss_diff_file = f"exp-fl-2_loss_diff_n{n}_{op}_{cor_mis}_weight_bl.npy"
    elif fl_method == "random":
        loss_diff_file = f"exp-fl-1_loss_diff_n{n}_{op}_{cor_mis}_weight_random.npy"
    else:
        raise ValueError(f"Unknown fl_method: {fl_method}")
    loss_diff_path = os.path.join(loss_diff_dir, loss_diff_file)
    return loss_diff_path

def get_true_prob_diff_path(n, w_num, beta, fl_method, loss_diff_dir, op, cor_mis):
    assert cor_mis in ["cor", "mis"], f"Unknown cor_mis: {cor_mis}"
    if fl_method == "ours":
        loss_diff_file = f"exp-fl-7_true_prob_diff_n{n}_beta{beta}_{op}_{cor_mis}_weight_ours.npy"
    elif fl_method == "bl":
        loss_diff_file = f"exp-fl-2_true_prob_diff_n{n}_{op}_{cor_mis}_weight_bl.npy"
    elif fl_method == "random":
        loss_diff_file = f"exp-fl-1_true_prob_diff_n{n}_{op}_{cor_mis}_weight_random.npy"
    else:
        raise ValueError(f"Unknown fl_method: {fl_method}")
    loss_diff_path = os.path.join(loss_diff_dir, loss_diff_file)
    return loss_diff_path


def main(fl_method, n, w_num, rank, beta):
    # rank を tgt_rank として利用（int型）
    tgt_rank = rank
    tgt_pos = ViTExperiment.CLS_IDX
    
    # プリトレーニング済みモデルとキャッシュの hidden states のロード
    pretrained_dir = ViTExperiment.c100.OUTPUT_DIR.format(k=0)
    batch_size = ViTExperiment.BATCH_SIZE
    ds_dir = os.path.join(ViTExperiment.DATASET_DIR, "c100_fold0")
    ds = load_from_disk(ds_dir)
    tgt_split = "repair"
    label_col = "fine_label"
    true_labels = np.array(ds[tgt_split][label_col])
    
    model = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
    model.eval()
    vit_from_last_layer = ViTFromLastLayer(model)
    vit_from_last_layer.eval()
    
    # キャッシュされた hidden states のロード（全サンプル対象）
    cache_path = os.path.join(pretrained_dir, "cache_hidden_states_before_layernorm_repair", "hidden_states_before_layernorm_11.npy")
    clean_hs = get_batched_hs(cache_path, batch_size)
    
    # Baseline 推論
    all_logits = []
    all_proba = []
    all_pred_labels = []
    for cached_state in tqdm(clean_hs, total=len(clean_hs), desc="Baseline"):
        logits = vit_from_last_layer(hidden_states_before_layernorm=cached_state, tgt_pos=tgt_pos)
        proba = torch.nn.functional.softmax(logits, dim=-1)
        all_logits.append(logits.detach().cpu().numpy())
        all_proba.append(proba.detach().cpu().numpy())
    all_logits = np.concatenate(all_logits, axis=0)
    all_proba = np.concatenate(all_proba, axis=0)
    all_pred_labels = all_logits.argmax(axis=-1)
    
    # Baseline Accuracy
    is_correct = (true_labels == all_pred_labels).astype(np.int32)
    total_correct = np.sum(is_correct)
    total = len(true_labels)
    accuracy_before = total_correct / total
    print(f"Baseline Accuracy: {accuracy_before:.4f} ({total_correct}/{total})")
    
    # Baseline Loss (各サンプルごと)
    logits_tensor = torch.from_numpy(all_logits)
    labels_tensor = torch.from_numpy(true_labels)
    criterion = nn.CrossEntropyLoss(reduction="none")
    loss_all = criterion(logits_tensor, labels_tensor).detach().cpu().numpy()
    
    # 実験条件のループ（misclf_type と fpfn は内部ループ）
    misclf_type_list = ["src_tgt", "tgt"] # allは対象外にする
    fpfn_list = [None, "fp", "fn"]
    results_list = []
    op_list = ["enhance", "suppress", "multiply-2"]
    
    for misclf_type in misclf_type_list:
        for fpfn in fpfn_list:
            # ルール：misclf_type=="all" は tgt_rank が 1 かつ fpfn が None のみ有効
            if misclf_type == "all":
                if tgt_rank >= 2 or fpfn is not None:
                    continue
            # misclf_type=="src_tgt" の場合、fpfn は None のみ
            if misclf_type == "src_tgt" and fpfn is not None:
                continue
            
            misclf_type_name = misclf_type if misclf_type != "tgt" or fpfn is None else f"tgt_{fpfn}"
            
            print(f"\nCondition: misclf_type={misclf_type}, tgt_rank={tgt_rank}, fpfn={fpfn}")
            misclf_pair, tgt_label, tgt_mis_indices = identfy_tgt_misclf(
                os.path.join(pretrained_dir, "misclf_info"),
                tgt_split="repair",
                misclf_type=misclf_type,
                tgt_rank=tgt_rank,
                fpfn=fpfn
            )
            print(f"misclf_pair: {misclf_pair}, tgt_label: {tgt_label}, # incorrect: {len(tgt_mis_indices)}")
            
            correct_indices = np.where(is_correct == 1)[0]
            incorrect_indices = tgt_mis_indices
            
            # Baseline の信頼度差分計算
            correct_proba = all_proba[correct_indices]
            incorrect_proba = all_proba[incorrect_indices]
            
            true_conf_corr = correct_proba[np.arange(len(correct_proba)), true_labels[correct_indices]] # 正解時の正解ラベルに対する予測確率
            true_conf_incorr = incorrect_proba[np.arange(len(incorrect_proba)), true_labels[incorrect_indices]] # 誤認識時の正解ラベルに対する予測確率
            
            # Baseline Loss metrics
            correct_loss = loss_all[correct_indices]
            incorrect_loss = loss_all[incorrect_indices]
            mean_loss_corr = np.mean(correct_loss)
            std_loss_corr = np.std(correct_loss)
            mean_loss_incorr = np.mean(incorrect_loss)
            std_loss_incorr = np.std(incorrect_loss)
            
            print(f"correct_loss.shape: {correct_loss.shape}, incorrect_loss.shape: {incorrect_loss.shape}")
            
            # --- 重み操作実験 ---
            location_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type_name}_weights_location")
            location_path = get_location_path(n, w_num, beta, fl_method, location_dir)
            pos_before, pos_after = np.load(location_path, allow_pickle=True)
            print(f"Location file: {location_path}")
            # print(f"pos_before.shape: {pos_before.shape}, pos_after.shape: {pos_after.shape}")
            print("pos_before.shape:", np.array(pos_before).shape)
            print("pos_after.shape:", np.array(pos_after).shape)

            # サンプルごとのロスの保存dir
            loss_diff_dir = os.path.join(location_dir, "loss_diff_per_sample")
            os.makedirs(loss_diff_dir, exist_ok=True)
            true_prob_diff_dir = os.path.join(location_dir, "true_prob_diff_per_sample")
            os.makedirs(true_prob_diff_dir, exist_ok=True)

            op_metrics = {}
            for op in op_list:
                if "multiply" in op:
                    # "multiply" を含む場合はその係数を取り出す
                    op_coeff = int(op.split("multiply")[-1])
                else:
                    op_coeff = op
                # opかけたモデルのロス - original modelのロスの差を保存するパス
                cor_loss_diff_path = get_loss_diff_path(n, w_num, beta, fl_method, loss_diff_dir, op, "cor")
                mis_loss_diff_path = get_loss_diff_path(n, w_num, beta, fl_method, loss_diff_dir, op, "mis")
                true_conf_corr_diff_path = get_true_prob_diff_path(n, w_num, beta, fl_method, true_prob_diff_dir, op, "cor")
                true_conf_incorr_diff_path = get_true_prob_diff_path(n, w_num, beta, fl_method, true_prob_diff_dir, op, "mis")
                # モデルのコピーで初期状態から再構築
                model_copy = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
                vit_from_last_layer_mod = ViTFromLastLayer(model_copy)
                vit_from_last_layer_mod.eval()
                dummy_in = [0] * (len(pos_before) + len(pos_after))
                set_new_weights(dummy_in, pos_before, pos_after, vit_from_last_layer_mod, op=op_coeff)
                _ = vit_from_last_layer_mod(hidden_states_before_layernorm=clean_hs[0], tgt_pos=tgt_pos)
                
                all_logits_mod = []
                all_proba_mod = []
                for cached_state in tqdm(clean_hs, total=len(clean_hs), desc=f"Modified op: {op}"):
                    logits_mod = vit_from_last_layer_mod(hidden_states_before_layernorm=cached_state, tgt_pos=tgt_pos)
                    proba_mod = torch.nn.functional.softmax(logits_mod, dim=-1)
                    all_logits_mod.append(logits_mod.detach().cpu().numpy())
                    all_proba_mod.append(proba_mod.detach().cpu().numpy())
                all_logits_mod = np.concatenate(all_logits_mod, axis=0)
                all_proba_mod = np.concatenate(all_proba_mod, axis=0)
                all_pred_labels_mod = all_logits_mod.argmax(axis=-1)
                acc_mod = np.sum(true_labels == all_pred_labels_mod) / len(true_labels)
                
                # Modified の信頼度差分計算
                correct_proba_mod = all_proba_mod[correct_indices]
                incorrect_proba_mod = all_proba_mod[incorrect_indices]
                
                true_conf_corr_mod = correct_proba_mod[np.arange(len(correct_proba_mod)), true_labels[correct_indices]]
                true_conf_incorr_mod = incorrect_proba_mod[np.arange(len(incorrect_proba_mod)), true_labels[incorrect_indices]]
                
                # Modified Loss metrics
                logits_tensor_mod = torch.from_numpy(all_logits_mod)
                loss_all_mod = criterion(logits_tensor_mod, labels_tensor).detach().cpu().numpy()
                correct_loss_mod = loss_all_mod[correct_indices]
                incorrect_loss_mod = loss_all_mod[incorrect_indices]
                mean_loss_corr_mod = np.mean(correct_loss_mod)
                std_loss_corr_mod = np.std(correct_loss_mod)
                mean_loss_incorr_mod = np.mean(incorrect_loss_mod)
                std_loss_incorr_mod = np.std(incorrect_loss_mod)
                
                op_metrics[op] = {
                    "mean_loss_correct_mod": mean_loss_corr_mod,
                    "std_loss_correct_mod": std_loss_corr_mod,
                    "mean_loss_incorrect_mod": mean_loss_incorr_mod,
                    "std_loss_incorrect_mod": std_loss_incorr_mod,
                }
                
                # ロスの差分を計算して保存
                assert len(correct_loss) == len(correct_loss_mod), "Length mismatch between original and modified loss arrays"
                assert len(incorrect_loss) == len(incorrect_loss_mod), "Length mismatch between original and modified loss arrays"
                cor_loss_diff = correct_loss_mod - correct_loss
                mis_loss_diff = incorrect_loss_mod - incorrect_loss
                print(f"cor_loss_diff.shape: {cor_loss_diff.shape}, mis_loss_diff.shape: {mis_loss_diff.shape}")
                np.save(cor_loss_diff_path, cor_loss_diff)
                np.save(mis_loss_diff_path, mis_loss_diff)
                print(f"Saved loss diff for {op} operation: {cor_loss_diff_path}, {mis_loss_diff_path}")
                
                # 正解への予測確率の差分を計算して保存
                assert len(true_conf_corr) == len(true_conf_corr_mod), "Length mismatch between original and modified confidence arrays"
                assert len(true_conf_incorr) == len(true_conf_incorr_mod), "Length mismatch between original and modified confidence arrays"
                true_conf_corr_diff = true_conf_corr_mod - true_conf_corr
                true_conf_incorr_diff = true_conf_incorr_mod - true_conf_incorr
                print(f"true_conf_corr_diff.shape: {true_conf_corr_diff.shape}, true_conf_incorr_diff.shape: {true_conf_incorr_diff.shape}")
                np.save(true_conf_corr_diff_path, true_conf_corr_diff)
                np.save(true_conf_incorr_diff_path, true_conf_incorr_diff)
                print(f"Saved confidence diff for {op} operation: {true_conf_corr_diff_path}, {true_conf_incorr_diff_path}")
                
            # --- 結果エントリ作成 ---
            result_entry = {
                "misclf_type": misclf_type,
                "tgt_rank": tgt_rank,
                "fpfn": fpfn,
                "method": fl_method,
                "n_ratio": n,
                "w_num": w_num,
                "beta": beta,
                "mean_loss_correct_baseline": mean_loss_corr,
                "std_loss_correct_baseline": std_loss_corr,
                "mean_loss_incorrect_baseline": mean_loss_incorr,
                "std_loss_incorrect_baseline": std_loss_incorr,
            }
            for op in op_list:
                result_entry[f"mean_loss_correct_mod_{op}"] = op_metrics[op]["mean_loss_correct_mod"]
                result_entry[f"std_loss_correct_mod_{op}"] = op_metrics[op]["std_loss_correct_mod"]
                result_entry[f"mean_loss_incorrect_mod_{op}"] = op_metrics[op]["mean_loss_incorrect_mod"]
                result_entry[f"std_loss_incorrect_mod_{op}"] = op_metrics[op]["std_loss_incorrect_mod"]
            
            results_list.append(result_entry)
    
    # 結果を DataFrame 化
    df_results = pd.DataFrame(results_list)
    return df_results

if __name__ == "__main__":
    all_results = []
    fl_method_list = ["ours"]
    # tgt_rank_list から各値を取り出して main に渡す
    tgt_rank_list = [1, 2, 3, 4, 5]
    beta_list = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    for tgt_rank in tgt_rank_list:
        for fl_method in fl_method_list:
            # exp_list = [ExperimentRepair1, Experiment3] # n_ratio = 24, 96
            # exp_list = [ExperimentRepair2] # n_ratio = 48
            exp_list = [Experiment4] # n_ratio = 12
            for exp in exp_list:
                n_ratio = exp.NUM_IDENTIFIED_WEIGHTS
                w_num = None
                for beta in beta_list:
                    print(f"\nfl_method: {fl_method}, n_ratio: {n_ratio}, w_num: {w_num}, tgt_rank: {tgt_rank}")
                    print(f"========================")
                    df_tmp = main(fl_method=fl_method, n=n_ratio, w_num=w_num, rank=tgt_rank, beta=beta)
                    all_results.append(df_tmp)
    df_all = pd.concat(all_results, ignore_index=True)
    unique_ns = df_all["n_ratio"].unique()
    n_str = '_'.join(map(str, unique_ns))
    output_csv_all = os.path.join(os.getcwd(), f"exp-fl-7-2_n{n_str}.csv")
    df_all.to_csv(output_csv_all, index=False)
    print(f"Aggregated results saved to: {output_csv_all}")
