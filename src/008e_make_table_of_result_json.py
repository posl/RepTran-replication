import argparse, os
from utils.constant import ViTExperiment
from utils.helper import json2dict
from itertools import product
import numpy as np

if __name__ == "__main__":
    # データセットをargparseで受け取る
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str)
    parser.add_argument('k', type=int, help="the fold id (0 to K-1)")
    parser.add_argument('misclf_type', type=str, help="the type of misclassification (src_tgt or tgt or all)", default="tgt")
    parser.add_argument("mode", type=str, help="the mode of the experiment (repair or retrain)", choices=["repair", "retrain"])
    parser.add_argument('--tgt_rank', type=int, help="the rank of the target misclassification type", default=None)
    args = parser.parse_args()
    ds_name = args.ds
    k = args.k
    tgt_rank = args.tgt_rank
    misclf_type = args.misclf_type
    mode = args.mode
    print(f"ds_name: {ds_name}, fold_id: {k}, tgt_rank: {tgt_rank}, misclf_type: {misclf_type}, mode: {mode}")
    
    # pretrained modelのディレクトリ
    pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR.format(k=k)
    # 結果とかログの保存先を先に作っておく
    if mode == "repair":
        # overall repair
        if misclf_type == "all":
            save_dir = os.path.join(pretrained_dir, f"all_repair_weight_by_de")
        # src_tgt or tgt repair
        else:
            save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_repair_weight_by_de")
        # n_listとalpha_listとtgt_splitを作成
        n_list = [5, 77, 109]
        # n_list = [77]
        alpha_list = [0.2, 0.4, 0.6, 0.8]
        tgt_split = ["repair", "test"]
        # 結果を保存するarr
        if misclf_type == "tgt":
            metrics_list = ["delta_f1_tgt", "repair_rate_tgt", "repair_rate_overall", "break_rate_overall", "delta_acc"]
        elif misclf_type == "src_tgt":
            metrics_list = ["repair_rate_tgt", "repair_rate_overall", "break_rate_overall", "delta_acc"]
        result_arr = np.zeros((len(n_list) * len(alpha_list), len(metrics_list) * len(tgt_split)))
        misclf_cnt_arr = np.zeros((len(n_list) * len(alpha_list), 2 * len(tgt_split)))
        for i, (n, alpha) in enumerate(product(n_list, alpha_list)):
            setting_id = f"n{n}_alpha{alpha}"
            result_row = []
            misclf_row = []
            for split in tgt_split:
                json_path = os.path.join(save_dir, f"{split}_metrics_for_repair_{setting_id}.json")
                metrics_dict = json2dict(json_path)
                for metric in metrics_list:
                    result_row.append(metrics_dict[metric])
                misclf_row.append(metrics_dict["tgt_misclf_cnt_new"])
                misclf_row.append(metrics_dict["new_injected_faults"])
            result_arr[i] = result_row
            misclf_cnt_arr[i] = misclf_row
        # result_arrをcsvで保存
        result_path = os.path.join(save_dir, "repair_result.csv")
        misclf_cnt_path = os.path.join(save_dir, "misclf_cnt.csv")
        np.savetxt(result_path, result_arr, delimiter=",", fmt="%.6f")
        np.savetxt(misclf_cnt_path, misclf_cnt_arr, delimiter=",", fmt="%d")
    elif mode == "retrain":
        # overall retrain
        if misclf_type == "all":
            save_dir = os.path.join(pretrained_dir, f"retraining_with_repair_set")
        # src_tgt or tgt retrain
        else:
            save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_retraining_with_only_repair_target")
    else:
        raise ValueError(f"mode {mode} is not supported")