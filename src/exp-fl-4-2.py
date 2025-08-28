import os
import numpy as np
import pandas as pd
from itertools import product
from collections import defaultdict
from utils.constant import ViTExperiment
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")

if __name__ == "__main__":
    ds = "c100"
    true_labels = range(100) if ds == "c100" else None
    tgt_split = "repair"
    exp_fl_1_save_path = f"./exp-fl-1_{ds}_proba_diff.csv"
    exp_fl_2_save_path = f"./exp-fl-2_{ds}_proba_diff.csv"
    exp_fl_3_save_path = f"./exp-fl-3_{ds}_proba_diff.csv"

    df1 = pd.read_csv(exp_fl_1_save_path)
    df2 = pd.read_csv(exp_fl_2_save_path)
    df3 = pd.read_csv(exp_fl_3_save_path)
    print(f"df1.shape: {df1.shape}")
    print(f"df2.shape: {df2.shape}")
    print(f"df3.shape: {df3.shape}")

    # df1,2,3を縦に結合
    ori_df = pd.concat([df1, df2, df3], ignore_index=True)
    print(f"df.shape: {ori_df.shape}")

    # diff_proba以外のユニークな値を表示 (デバッグ用)
    for col in ori_df.columns:
        if col not in ["diff_proba"]:
            print(f"{col}: {ori_df[col].unique()}")
            
    # num_weight のユニークな値ごとに処理
    for wnum in ori_df["num_weight"].unique():
        df = ori_df[ori_df["num_weight"] == wnum]
        print(f"wnum: {wnum}, len(df): {len(df)}")

        #========================================================================
        # 1) (fl_method, fl_target) の行Index / misclf_type の列Index を準備
        #========================================================================
        method_target_pairs = (
            df[["fl_target", "fl_method"]]
            .drop_duplicates()
            .sort_values(["fl_target", "fl_method"])
            .apply(tuple, axis=1)
            .tolist()
        )
        row_index = pd.MultiIndex.from_tuples(
            method_target_pairs, names=["fl_target", "fl_method"]
        )

        # スクリプト2で使っていた5種類
        known_misclf_types = ["all", "src_tgt", "tgt_all", "tgt_fp", "tgt_fn"]
        # known_misclf_types = ["src_tgt", "tgt_all", "tgt_fp", "tgt_fn"]
        # 1次元だけのカラム
        col_index = pd.Index(known_misclf_types, name="misclf_type")

        # 結果テーブル (行=(fl_method,fl_target), 列=misclf_type)
        result_table = pd.DataFrame(index=row_index, columns=col_index, data=-1)

        #========================================================================
        # 2) misclf_type と fpfn の組み合わせを取得してループ
        #    ただし rank は一切絞らず、全 rank を対象に集計
        #========================================================================
        misclf_type_fpfn_list = df[["misclf_type", "fpfn"]].drop_duplicates().values.tolist()
        print("misclf_type_fpfn_list:", misclf_type_fpfn_list)

        for (mc_type, fpfn_val) in misclf_type_fpfn_list:
            # 例: mc_type='all', fpfn_val=nan
            #     mc_type='tgt', fpfn_val='fp'
            #     mc_type='src_tgt', fpfn_val=nan
            #     etc.

            # どの列名に書き込むかを決める (script2 のロジック)
            # mc_type='all', 'src_tgt', 'tgt', ...
            # fpfn_val= NaN / 'fp' / 'fn'
            # ここで 'tgt' なら 'tgt_all' / 'tgt_fp' / 'tgt_fn' に振り分け
            col_name = None
            if mc_type == "all":
                # 'all' は rank関係なく
                # 書き込み先は known_misclf_types の 'all'
                col_name = "all"
            elif mc_type == "src_tgt":
                col_name = "src_tgt"
            elif mc_type == "tgt":
                # 'tgt' + fp/fn のパターン
                if pd.isna(fpfn_val):
                    col_name = "tgt_all"
                elif fpfn_val == "fp":
                    col_name = "tgt_fp"
                elif fpfn_val == "fn":
                    col_name = "tgt_fn"
                else:
                    continue  # 不明なパターンはスキップ
            else:
                # もし 'tgt_all' とか既に入ってる場合があれば対応
                # あるいは余計なものはスキップ
                continue

            if col_name not in known_misclf_types:
                # 既定の5種に含まれないならスキップ
                continue

            print(f"\n==== Collecting for mc_type={mc_type}, fpfn_val={fpfn_val} => col={col_name}")

            #====================================================================
            # 3) データ抽出 (rankを無視: 全rank) + misclf_type/fpfn でフィルタ
            #====================================================================
            # df_extracted = df.copy()  # 全rank含む
            # -> ただし mc_type == 'all'なら何も絞らない (fpfn_val があるならエラー?)
            # -> mc_type == 'src_tgt'なら df_extracted = df[df["misclf_type"]=='src_tgt']
            # -> mc_type == 'tgt' なら df_extracted = df[df["misclf_type"]=='tgt'] etc.

            if mc_type == "all":
                # all は misclf_type を絞らず
                if not pd.isna(fpfn_val):
                    # all なのに fpfn_val があるのは整合しないはず → スキップかcontinue
                    print("all + fpfn_val != nan → skip")
                    continue
                df_extracted = df[df["misclf_type"] == mc_type]
            else:
                df_extracted = df[df["misclf_type"] == mc_type]
                if pd.isna(fpfn_val):
                    df_extracted = df_extracted[df_extracted["fpfn"].isna()]
                else:
                    df_extracted = df_extracted[df_extracted["fpfn"] == fpfn_val]

            if len(df_extracted) == 0:
                print("No data in df_extracted => skip")

            # df_extractedの各列のユニーク値を表示
            # for col in df_extracted.columns:
            #     if col not in ["diff_proba"]:
            #         print(f"{col}: {df_extracted[col].unique()}")
            #         print(f"{col}: {df_extracted[col].value_counts()}")
            # df_extractedの 'fl_method' 列のユニーク値ごとに，他の列のユニーク値を表示
            # for method_val in df_extracted["fl_method"].unique():
            #     print(f"fl_method={method_val} ({len(df_extracted[df_extracted['fl_method'] == method_val])})")
            #     for col in df_extracted.columns:
            #         if col not in ["diff_proba"]:
                        # print(f"{col}: {df_extracted[df_extracted['fl_method'] == method_val][col].unique()}")
                        # print(df_extracted[df_extracted['fl_method'] == method_val][col].value_counts())
            
            # df_extractedの 'n' 列のユニーク値ごとに，他の列のユニーク値を表示
            # for n_val in df_extracted["n"].unique():
            #     print(f"n={n_val} ({len(df_extracted[df_extracted['n'] == n_val])})")
            #     for col in df_extracted.columns:
            #         if col not in ["diff_proba"]:
            #             # print(f"{col}: {df_extracted[df_extracted['n'] == n_val][col].unique()}")
            #             print(df_extracted[df_extracted['n'] == n_val][col].value_counts())

            #====================================================================
            # 4) groupby(["label", "op", "fl_method", "fl_target"]) で mean_diff_proba
            #====================================================================
            # print(df_extracted.columns)
            grouped_df = (
                df_extracted
                .groupby(["label", "op", "fl_method", "fl_target"], as_index=False)
                .agg(
                    count_rows=("diff_proba", "count"),
                    mean_diff_proba=("diff_proba", "mean"),
                    std_diff_proba=("diff_proba", "std"))
            )
            
            # print(f"count_rows: {grouped_df['count_rows'].sum()}")
            # print(f"grouped_df.shape: {grouped_df.shape}")
            grouped_df["mean_diff_proba"] = grouped_df["mean_diff_proba"] * 100

            #====================================================================
            # 5) (fl_method, fl_target) ごとに「ラベルごと最大の行」を取り、>0 のクラスをカウント
            #====================================================================
            for (method_val, target_val), sub_df in grouped_df.groupby(["fl_method", "fl_target"]):
                print(method_val, target_val, f"({len(sub_df)})")
                # print(f"sub_df.shape: {sub_df.shape}")
                # print(f"sub_df: {sub_df}")
                if len(sub_df) == 0:
                    continue
                
                # ラベルごとに最大
                sub_max = sub_df.groupby("label").apply(
                    lambda g: g.loc[g["mean_diff_proba"].idxmax()]
                )
                n_pos = (sub_max["mean_diff_proba"] > 0).sum()

                # テーブルに書き込み (加算 or 上書き)
                # 今回は上書き想定 => もし別のfpfnで同じセルに加算したくない場合
                # → result_table.loc[(method_val, target_val), col_name] = n_pos
                #
                # もし複数fpfnを合算したいなら "+=" に書き換える
                #
                # current_val = result_table.loc[(method_val, target_val), col_name]
                # 上書き or 最大値比較 or 加算など、方針に合わせてどうぞ
                # 例: 上書き => 
                # if n_pos > current_val:
                result_table.loc[(target_val, method_val), col_name] = n_pos

        #========================================================================
        # 6) テーブルをSave (列は misclf_type だけ)
        #========================================================================
        print("\n===== Final Table: (fl_method, fl_target) x misclf_type =====")
        print(result_table)

        out_csv = f"./exp-fl-4-2_{ds}_no_rank_table_wnum{wnum}.csv"
        desired_index_order = [
            ("neuron", "random"),
            ("weight", "random"),
            ("neuron", "ig"),
            ("weight", "bl"),
            ("neuron", "vdiff"),
            ("weight", "vdiff"),
            ("neuron", "vdiff+mean_act"),
            ("weight", "vdiff+mean_act+grad"),
        ]
        # 再インデックス (存在しないペアがある場合は NaN になります)
        result_table = result_table.reindex(desired_index_order)
        result_table.to_csv(out_csv)
        print(f"Saved table to {out_csv}")
