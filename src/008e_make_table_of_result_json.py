import argparse, os
from utils.constant import ViTExperiment
from utils.helper import json2dict
from itertools import product
import numpy as np

if __name__ == "__main__":
    # Accept dataset via argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str)
    parser.add_argument('k', type=int, help="the fold id (0 to K-1)")
    parser.add_argument('misclf_type', type=str, help="the type of misclassification (src_tgt or tgt or all)", default="tgt")
    parser.add_argument("mode", type=str, help="the mode of the experiment (repair or retrain)", choices=["repair", "retrain"])
    parser.add_argument('--tgt_rank', type=int, help="the rank of the target misclassification type", default=1)
    parser.add_argument("--fpfn", type=str, help="the type of misclassification (fp or fn)", default=None, choices=["fp", "fn"])
    args = parser.parse_args()
    ds_name = args.ds
    k = args.k
    tgt_rank = args.tgt_rank
    misclf_type = args.misclf_type
    mode = args.mode
    fpfn = args.fpfn
    print(f"ds_name: {ds_name}, fold_id: {k}, tgt_rank: {tgt_rank}, misclf_type: {misclf_type}, mode: {mode}, fpfn: {fpfn}")
    
    # Directory for pretrained model
    pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR.format(k=k)
    # Create save directories for results and logs in advance
    if mode == "repair":
        # overall repair
        if misclf_type == "all":
            save_dir = os.path.join(pretrained_dir, f"all_repair_weight_by_de")
            retrain_dir = os.path.join(pretrained_dir, f"retraining_with_repair_set")
            used_met = "f1"
        # src_tgt or tgt repair
        else:
            save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_repair_weight_by_de")
            retrain_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_retraining_with_only_repair_target")
            if fpfn is not None:
                save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_{fpfn}_repair_weight_by_de")
                retrain_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_{fpfn}_retraining_with_only_repair_target")
                used_met = "recall" if fpfn == "fn" else "precision"
        # n_listとalpha_listとtgt_splitを作成
        n_list = [5, 77, 109]
        # n_list = [77]
        alpha_list = [0.2, 0.4, 0.6, 0.8]
        tgt_split = ["repair", "test"]
        bounds_list = [None, "Arachne", "ContrRep"]
        # 結果をSaveするarr
        if misclf_type == "tgt":
            metrics_list = [f"{used_met}_tgt_old", f"{used_met}_tgt_new", f"delta_{used_met}_tgt", "repair_rate_tgt", "repair_rate_overall", "break_rate_overall", "delta_acc"]
        elif misclf_type == "src_tgt":
            metrics_list = ["repair_rate_tgt", "repair_rate_overall", "break_rate_overall", "delta_acc"]
        result_arr = np.zeros((len(n_list) * len(alpha_list) * len(bounds_list) + 1,  len(metrics_list) * len(tgt_split)))
        misclf_cnt_arr = np.zeros((len(n_list) * len(alpha_list) * len(bounds_list) + 1, 2 * len(tgt_split)))
        for i, (bounds, n, alpha) in enumerate(product(bounds_list, n_list, alpha_list)):
            setting_id = f"n{n}_alpha{alpha}_bounds{bounds}" if bounds is not None else f"n{n}_alpha{alpha}"
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
        # 最終行にはretrainの結果を追加
        retrain_result_row = []
        retrain_misclf_row = []
        for split in tgt_split:
            retrain_metrics_dict = json2dict(os.path.join(retrain_dir, f"{split}_metrics_for_repair_{misclf_type}.json"))
            for metric in metrics_list:
                retrain_result_row.append(retrain_metrics_dict[metric])
            retrain_misclf_row.append(retrain_metrics_dict["tgt_misclf_cnt_new"])
            retrain_misclf_row.append(retrain_metrics_dict["new_injected_faults"])
        result_arr[-1] = retrain_result_row
        misclf_cnt_arr[-1] = retrain_misclf_row
        # result_arrをcsvでSave
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