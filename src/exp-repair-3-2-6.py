import os
import json
import pandas as pd
from pandas.api.types import is_float_dtype
from collections import defaultdict
from utils.constant import ViTExperiment
from itertools import product

if __name__ == "__main__":
    # 変数の定義
    ds = "c100"
    k = 0
    tgt_rank_list = [1, 2, 3]
    misclf_type_list = ["src_tgt", "tgt"]
    fpfn_list = [None, "fp", "fn"]
    tgt_split_list = ["repair", "test"]
    num_reps = 5
    alpha = 10 / 11
    fl_method_list = ["ours", "random"]  # 追加: random methodも試す場合
    setting_id = f"alpha{alpha}_boundsArachne"

    # 保存先ディレクトリ（仮定）
    pretrained_dir = getattr(ViTExperiment, ds).OUTPUT_DIR.format(k=k)

    
    for fl_method in fl_method_list:
        # 結果の格納用
        results = []
        
        for tgt_rank, misclf_type, fpfn in product(tgt_rank_list, misclf_type_list, fpfn_list):
            if (misclf_type in ["src_tgt", "all"] and fpfn is not None) or (misclf_type == "tgt" and fpfn is None):
                continue

            misclf_ptn = misclf_type if fpfn is None else f"{misclf_type}_{fpfn}"
            save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_ptn}_repair_weight_by_de")
            row = {"ds": ds, "tgt_rank": tgt_rank, "misclf_type": misclf_ptn, "fl_method": fl_method}

            for tgt_split in tgt_split_list:
                rr_list, br_list, racc_list, diff_corr_list, t_repair_list = [], [], [], [], []
                for reps_id in range(num_reps):
                    filename = f"exp-repair-3-2-metrics_for_{tgt_split}_{setting_id}_{fl_method}_reps{reps_id}.json"
                    json_path = os.path.join(save_dir, filename)
                    if not os.path.exists(json_path):
                        raise FileNotFoundError(f"JSON file not found: {json_path}")
                    with open(json_path, "r") as f:
                        d = json.load(f)
                    rr_list.append(d.get("repair_rate_tgt"))
                    br_list.append(d.get("break_rate_tgt"))
                    racc_list.append(d.get("r_acc"))
                    diff_corr_list.append(d.get("diff_correct"))
                    if tgt_split == "repair":
                        t_repair_list.append(d.get("tot_time"))

                row[f"RR_{tgt_split}"] = sum(rr_list)/len(rr_list)
                row[f"BR_{tgt_split}"] = sum(br_list)/len(br_list)
                row[f"Racc_{tgt_split} (#diff)"] = f"{sum(racc_list)/len(racc_list):.3f} ({sum(diff_corr_list)/len(diff_corr_list):.1f})"
                if tgt_split == "repair":
                    fl_time_path = "/src/src/exp-repair-3-1-1_time_pareto.csv"
                    df_fl_time = pd.read_csv(fl_time_path)
                    fpfn_match = "" if fpfn is None else fpfn
                    matched_row = df_fl_time[
                        (df_fl_time["ds"] == ds) &
                        (df_fl_time["k"] == k) &
                        (df_fl_time["tgt_rank"] == tgt_rank) &
                        (df_fl_time["misclf_type"] == misclf_type) &
                        (df_fl_time["fpfn"].fillna("") == fpfn_match)
                    ]
                    row["t_fl"] = matched_row["elapsed_time"].values[0]
                    row["t_repair"] = sum(t_repair_list)/len(t_repair_list)
        results.append(row)
        # データフレーム化
        df_flat = pd.DataFrame(results)
        # 小数表示のフォーマット設定（小数第3位）
        float_cols = [col for col in df_flat.columns if is_float_dtype(df_flat[col])]
        # 表示桁数を揃える（実体は変えず文字列化せず）
        df_flat[float_cols] = df_flat[float_cols].round(3)
        csv_path = f"./exp-repair-3-2-6-{setting_id}_{fl_method}.csv"
        # 実行時間列を最後に移動
        time_cols = ["t_fl", "t_repair"]
        other_cols = [col for col in df_flat.columns if col not in time_cols]
        df_flat = df_flat[other_cols + time_cols]
        df_flat.to_csv(csv_path, index=False, float_format="%.3f")