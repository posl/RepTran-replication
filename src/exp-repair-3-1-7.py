import os
import json
import pandas as pd
from pandas.api.types import is_float_dtype
from collections import defaultdict
from utils.constant import ViTExperiment
from glob import glob

if __name__ == "__main__":
    # 変数の定義
    # ds = "c100"  # データセット名
    ds_list = ["c100", "tiny-imagenet"]
    k = 0
    tgt_rank_list = [1, 2, 3]
    misclf_type_list = ["src_tgt", "tgt"]
    fpfn_list = [None, "fp", "fn"]
    tgt_split_list = ["repair", "test"]
    num_reps = 5
    alpha = 10 / 11
    fl_method = "bl"
    setting_id = f"alpha{alpha}_boundsArachne"

    for ds in ds_list:
        # 保存先ディレクトリ（仮定）
        exp_obj = getattr(ViTExperiment, ds.replace("-", "_"))
        pretrained_dir = exp_obj.OUTPUT_DIR.format(k=k)

        # 結果の格納用
        results = []

        # JSONファイルの探索と集計
        for tgt_rank in tgt_rank_list:
            for misclf_type in misclf_type_list:
                for fpfn in fpfn_list:
                    # 関係ない設定時はスキップ
                    if (misclf_type in ["src_tgt", "all"] and fpfn is not None) or (misclf_type == "tgt" and fpfn is None):
                        continue
                    misclf_ptn = misclf_type if fpfn is None else f"{misclf_type}_{fpfn}"
                    # jsonの保存されているディレクトリ
                    save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_ptn}_repair_weight_by_de")
                    row = {"ds": ds, "tgt_rank": tgt_rank, "misclf_type": misclf_ptn}
                    # repair or test
                    for tgt_split in tgt_split_list:
                        rr_list, br_list, racc_list, diff_corr_list = [], [], [], []
                        t_repair_list = []
                        for reps_id in range(num_reps):
                            # JSONファイルのパスを生成
                            filename = f"exp-repair-3-1-metrics_for_{tgt_split}_{setting_id}_{fl_method}_reps{reps_id}.json"
                            json_path = os.path.join(save_dir, filename)
                            if not os.path.exists(json_path):
                                raise FileNotFoundError(f"JSON file not found: {json_path}")
                            with open(json_path, "r") as f:
                                d = json.load(f)
                            rr_list.append(d.get("repair_rate_tgt"))
                            br_list.append(d.get("break_rate_overall"))
                            racc_list.append(d.get("r_acc"))
                            diff_corr_list.append(d.get("diff_correct"))
                            # tgt_splitが"repair"の時は実行時間の記録も行う
                            if tgt_split == "repair":
                                t_repair_list.append(d.get("tot_time"))
                        # 5回の平均を計算
                        row[f"RR_{tgt_split}"] = sum(rr_list)/len(rr_list)
                        row[f"BR_{tgt_split}"] = sum(br_list)/len(br_list)
                        avg_racc = sum(racc_list)/len(racc_list)
                        avg_diff = sum(diff_corr_list)/len(diff_corr_list)
                        fmt_col = f"Racc_{tgt_split} (#diff)"
                        row[fmt_col] = f"{avg_racc:.4f} ({avg_diff:.1f})"
                        if tgt_split == "repair":
                            # 念の為FL実行時間も記録
                            fl_time_path = f"/src/src/exp-repair-3-1-1_time_pareto_{ds}.csv"
                            df_fl_time = pd.read_csv(fl_time_path)
                            # fpfn=None のときは "" に置き換える（csv内の表記と合わせる）
                            fpfn_match = "" if fpfn is None else fpfn
                            matched_row = df_fl_time[
                                (df_fl_time["ds"] == ds) &
                                (df_fl_time["k"] == k) &
                                (df_fl_time["tgt_rank"] == tgt_rank) &
                                (df_fl_time["misclf_type"] == misclf_type) &
                                (df_fl_time["fpfn"].fillna("") == fpfn_match)
                            ]
                            row["t_fl"] = matched_row["elapsed_time"].values[0]
                            # repair実行時間の平均を計算
                            row["t_repair"] = sum(t_repair_list)/len(t_repair_list)
                    results.append(row)

        # データフレーム化
        df_flat = pd.DataFrame(results)
        # 小数表示のフォーマット設定（小数第3位）
        float_cols = [col for col in df_flat.columns if is_float_dtype(df_flat[col])]
        # 表示桁数を揃える（実体は変えず文字列化せず）
        df_flat[float_cols] = df_flat[float_cols].round(3)
        csv_path = f"./exp-repair-3-1-7-{setting_id}_{fl_method}_{ds}.csv"
        # 実行時間列を最後に移動
        time_cols = ["t_fl", "t_repair"]
        other_cols = [col for col in df_flat.columns if col not in time_cols]
        df_flat = df_flat[other_cols + time_cols]
        df_flat.to_csv(csv_path, index=False, float_format="%.4f")
        print(f"Results saved to {csv_path}")