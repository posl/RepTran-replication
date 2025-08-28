import os
import math
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
from collections import defaultdict
from utils.constant import ViTExperiment
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("ticks")


def main(
    ds_name,
    k,
    tgt_rank,
    n,
    misclf_type,
    fpfn,
    run_all=True,
    fl_methods=None,
    agg="abs_mean",
):
    # print(f"Dataset: {ds_name}, k: {k}, tgt_rank: {tgt_rank}, n: {n}, misclf_type: {misclf_type}, fpfn: {fpfn}")
    pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR.format(k=k)
    # FL情報のSave場所
    save_dir = os.path.join(
        pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_weights_location"
    )
    if misclf_type == "all":
        save_dir = os.path.join(pretrained_dir, f"all_weights_location")
    if fpfn is not None and misclf_type == "tgt":
        save_dir = os.path.join(
            pretrained_dir,
            f"misclf_top{tgt_rank}",
            f"{misclf_type}_{fpfn}_weights_location",
        )

    results = []

    # plt.figure(figsize=(8, 6))
    if fl_methods is None:
        for op in ["enh", "sup"]:
            for fl_method in ["vdiff", "random"]: # TODO: Knowledge Neuronsも追加
                proba_save_dir = os.path.join(save_dir, f"proba_n{n}_{fl_method}")
                # print(f"proba_save_dir: {proba_save_dir}")

                for tl in true_labels:
                    # オリジナルのモデルの予測確率を取得
                    ori_proba_save_path = os.path.join(
                        pretrained_dir, "pred_results", f"{tgt_split}_proba_{tl}.npy"
                    )
                    ori_proba = np.load(ori_proba_save_path)[
                        :, tl
                    ]  # tlへの予測確率 # 形状は (tlのラベルのサンプル数, ) の1次元配列

                    # 特定位置だけopで変更したモデルの予測確率を取得
                    proba_save_path = os.path.join(
                        proba_save_dir, f"{tgt_split}_proba_{op}_{tl}.npy"
                    )
                    proba = np.load(proba_save_path)[:, tl]  # tlへの予測確率
                    # print(f"Op: {op}, True label: {tl}, {tgt_split} proba shape: {proba.shape}, ori proba shape: {ori_proba.shape}")
                    assert (
                        proba.shape == ori_proba.shape
                    ), f"{proba.shape} != {ori_proba.shape}"
                    mean_diff = np.mean(
                        proba - ori_proba
                    )  # ラベルtlの全サンプルに対する予測確率の変化の平均

                    results.append(
                        {
                            "n": n,
                            "k": k,
                            "tgt_rank": tgt_rank,
                            "misclf_type": misclf_type,
                            "fpfn": fpfn,
                            "fl_method": fl_method,
                            "op": op,
                            "label": tl,
                            "diff_proba": mean_diff,
                        }
                    )
    elif isinstance(fl_methods, list):
        fl_method = "_".join(
            fl_methods
        )  # len(fl_methods)が1のときはfl_methods[0]そのまま
        proba_save_dir = os.path.join(save_dir, f"proba_n{n}_{fl_method}")

        if "vdiff_asc" in fl_method and "vdiff_desc" in fl_method:
            op_list = ["ae_ds", "as_de"]
        elif "vdiff_asc" in fl_method:
            op_list = ["ae", "as"]
        elif "vdiff_desc" in fl_method:
            op_list = ["de", "ds"]
        else:
            raise ValueError(f"Invalid fl_method: {fl_method}")

        for op in op_list:
            for tl in true_labels:
                # オリジナルのモデルの予測確率を取得
                ori_proba_save_path = os.path.join(
                    pretrained_dir, "pred_results", f"{tgt_split}_proba_{tl}.npy"
                )
                ori_proba = np.load(ori_proba_save_path)[
                    :, tl
                ]  # tlへの予測確率 # 形状は (tlのラベルのサンプル数, ) の1次元配列

                # 特定位置だけopで変更したモデルの予測確率を取得
                proba_save_path = os.path.join(
                    proba_save_dir, f"{tgt_split}_proba_{op}_{tl}.npy"
                )
                proba = np.load(proba_save_path)[:, tl]  # tlへの予測確率
                # print(f"Op: {op}, True label: {tl}, {tgt_split} proba shape: {proba.shape}, ori proba shape: {ori_proba.shape}")
                assert (
                    proba.shape == ori_proba.shape
                ), f"{proba.shape} != {ori_proba.shape}"
                mean_diff = np.mean(
                    proba - ori_proba
                )  # ラベルtlの全サンプルに対する予測確率の変化の平均

                results.append(
                    {
                        "n": n,
                        "k": k,
                        "tgt_rank": tgt_rank,
                        "misclf_type": misclf_type,
                        "fpfn": fpfn,
                        "fl_method": fl_method,
                        "op": op,
                        "diff_proba_mean": mean_diff,
                    }
                )

    return pd.DataFrame(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", nargs="?", type=str, default="c100", help="dataset name")
    parser.add_argument("--run_all", action="store_true", help="run all settings")
    parser.add_argument(
        "--run_asc_desc", action="store_true", help="run only for asc and desc settings"
    )
    parser.add_argument("--agg", type=str, help="aggregate results", default=None)
    args = parser.parse_args()
    ds = args.ds
    run_all = args.run_all
    run_asc_desc = args.run_asc_desc
    agg = args.agg

    # run_allとrun_asc_descが同時に指定されている場合はエラー
    assert not (
        run_all and run_asc_desc
    ), "run_all and run_asc_desc cannot be specified at the same time"

    if run_all:
        true_labels = range(100) if ds == "c100" else None
        tgt_split = "repair"
        k_list = range(5)
        tgt_rank_list = range(1, 4)
        n_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 77, 109]
        misclf_type_list = ["all", "src_tgt", "tgt"]
        fpfn_list = [None, "fp", "fn"]

        # DataFrame to store all results
        all_results = pd.DataFrame()

        for k, tgt_rank, n, misclf_type, fpfn in tqdm(
            product(k_list, tgt_rank_list, n_list, misclf_type_list, fpfn_list),
            total=len(k_list)
            * len(tgt_rank_list)
            * len(n_list)
            * len(misclf_type_list)
            * len(fpfn_list),
        ):
            if (misclf_type == "src_tgt" or misclf_type == "all") and fpfn is not None:
                continue
            # 各パラメータの組み合わせでmain関数から1行分のDataFrameを取得し、all_resultsに連結
            result_df = main(
                ds, k, tgt_rank, n, misclf_type, fpfn, run_all=run_all, agg=agg
            )
            all_results = pd.concat([all_results, result_df], ignore_index=True)
        # Save all_results
        save_path = f"./{ds}_proba_diff_{agg}_run_all.csv" if agg is not None else f"./{ds}_proba_diff_run_all.csv"
        all_results.to_csv(save_path, index=False)
    elif run_asc_desc:
        true_labels = range(100) if ds == "c100" else None
        tgt_split = "repair"
        k_list = range(5)
        tgt_rank_list = range(1, 4)
        org_n_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 77, 109]
        n_list = [round(n / math.sqrt(2)) for n in org_n_list]
        # fl_methods = ["vdiff_asc", "vdiff_desc"]
        fl_methods = ["vdiff_desc"]
        misclf_type_list = ["all", "src_tgt", "tgt"]
        fpfn_list = [None, "fp", "fn"]

        # DataFrame to store all results
        all_results = pd.DataFrame()

        for k, tgt_rank, n, misclf_type, fpfn in tqdm(
            product(k_list, tgt_rank_list, n_list, misclf_type_list, fpfn_list),
            total=len(k_list)
            * len(tgt_rank_list)
            * len(n_list)
            * len(misclf_type_list)
            * len(fpfn_list),
        ):
            if (misclf_type == "src_tgt" or misclf_type == "all") and fpfn is not None:
                continue
            # 各パラメータの組み合わせでmain関数から1行分のDataFrameを取得し、all_resultsに連結
            result_df = main(
                ds,
                k,
                tgt_rank,
                n,
                misclf_type,
                fpfn,
                run_all=run_all,
                fl_methods=fl_methods,
                agg=agg,
            )
            all_results = pd.concat([all_results, result_df], ignore_index=True)
        # Save all_results
        save_path = f"./{ds}_proba_diff_{agg}_run_all.csv" if agg is not None else f"./{ds}_proba_diff_run_all.csv"
        all_results.to_csv(save_path, index=False)
    else:
        raise ValueError("Please specify either --run_all or --run_asc_desc")
