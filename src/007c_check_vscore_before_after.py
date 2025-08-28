import os, sys, time, pickle, json, math, random
import argparse
from tqdm import tqdm
from collections import defaultdict
from itertools import product
import numpy as np
from utils.helper import get_device, json2dict
from utils.vit_util import localize_weights, localize_weights_random, identfy_tgt_misclf, get_vscore_diff_and_sim
from utils.constant import ViTExperiment
from utils.log import set_exp_logging
from logging import getLogger
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")


logger = getLogger("base_logger")

def main(ds_name, k, tgt_rank, n, fl_method, misclf_type, fpfn, run_all=False):
    print(f"ds_name: {ds_name}, fold_id: {k}, tgt_rank: {tgt_rank}, n: {n}, fl_method: {fl_method}, misclf_type: {misclf_type}, fpfn: {fpfn}")
    
    # pretrained modelのディレクトリ
    pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR.format(k=k)
    # 結果とかログの保存先を先に作っておく
    save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_weights_location")
    if misclf_type == "all":
        save_dir = os.path.join(pretrained_dir, f"all_weights_location")
    if fpfn is not None and misclf_type == "tgt":
        save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_{fpfn}_weights_location")
    os.makedirs(save_dir, exist_ok=True)
    # このpythonのファイル名を取得
    this_file_name = os.path.basename(__file__).split(".")[0]
    exp_name = f"{this_file_name}_n{n}" if not run_all else f"{this_file_name}_run_all"
    # loggerの設定をして設定情報を表示
    logger = set_exp_logging(exp_dir=save_dir, exp_name=exp_name)
    logger.info(f"ds_name: {ds_name}, fold_id: {k}, tgt_rank: {tgt_rank}, n: {n}, fl_method: {fl_method}, misclf_type: {misclf_type}")

    # tgt_rankの誤分類情報を取り出す
    tgt_split = "repair" # NOTE: we only use repair split for repairing
    logger.info(f"tgt_split: {tgt_split}")
    misclf_info_dir = os.path.join(pretrained_dir, "misclf_info")
    misclf_pair, tgt_label, tgt_mis_indices = identfy_tgt_misclf(misclf_info_dir, tgt_split=tgt_split, tgt_rank=tgt_rank, misclf_type=misclf_type, fpfn=fpfn)
    logger.info(f"misclf_pair: {misclf_pair}, tgt_label: {tgt_label}, len(tgt_mis_indices): {len(tgt_mis_indices)}")

    if misclf_type == "src_tgt" or misclf_type == "tgt":
        vscore_before_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", "vscores_before")
        vscore_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", "vscores")
        vscore_after_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", "vscores_after")
    elif misclf_type == "all":
        vscore_before_dir = os.path.join(pretrained_dir, "vscores_before")
        vscore_dir = os.path.join(pretrained_dir, "vscores")
        vscore_after_dir = os.path.join(pretrained_dir, "vscores_after")
    logger.info(f"vscore_before_dir: {vscore_before_dir}")
    logger.info(f"vscore_dir: {vscore_dir}")
    logger.info(f"vscore_after_dir: {vscore_after_dir}")
    # localizationを実行
    vdiff_dic = get_vscore_diff_and_sim(vscore_before_dir, vscore_dir, vscore_after_dir, tgt_split=tgt_split, misclf_pair=misclf_pair, tgt_label=tgt_label, fpfn=fpfn)
    return vdiff_dic
    
def aggregate_and_plot(result_dicts, seed_for_vdiff, name_vdiff_pick):
    # 集約用のデータ構造
    aggregated_data = defaultdict(lambda:
                                  defaultdict(lambda: 
                                              {"vdiff": {"before": [], "after": [], "intermediate": []}, 
                                                "cos_sim": {"before": [], "after": [], "intermediate": []}}
                                            )
                                )
    # データを集約
    for (k, tgt_rank, n, misclf_type, fpfn), vdiff_dic in result_dicts.items():
        if (misclf_type == "src_tgt" or misclf_type == "all") and fpfn is not None:
            continue
        for phase in ["before", "after", "intermediate"]:
            aggregated_data[misclf_type][fpfn]["vdiff"][phase].append(vdiff_dic[phase]["vdiff"])
            aggregated_data[misclf_type][fpfn]["cos_sim"][phase].append(vdiff_dic[phase]["cos_sim"])
    # 可視化
    for misclf_type, fpfn_data in aggregated_data.items():
        for fpfn, metrics in fpfn_data.items():
            if (misclf_type == "src_tgt" or misclf_type == "all") and fpfn is not None:
                continue
            mistype_str = "_".join([misclf_type, fpfn]) if fpfn is not None else misclf_type
            for metric, phase_data in metrics.items():
                if metric == "cos_sim":
                    cos_sim_save_dir = "cos_sim_plot"
                    os.makedirs(cos_sim_save_dir, exist_ok=True)
                    save_path = os.path.join(cos_sim_save_dir, f"cos_sim_{mistype_str}.png")
                    plt.figure(figsize=(10, 6))
                    for phase, vals in phase_data.items():
                        # 各レイヤごとの標準偏差を計算
                        stacked_cos_sim = np.stack(vals)  # 全データをスタック
                        mean_cos_sim = np.mean(stacked_cos_sim, axis=0)  # 平均値
                        std_cos_sim = np.std(stacked_cos_sim, axis=0)  # 標準偏差

                        # phaseごとに3色を使い分ける
                        color = "blue" if phase == "before" else "red" if phase == "after" else "green"

                        # プロット
                        layers = np.arange(len(mean_cos_sim))  # レイヤID（x軸）
                        plt.plot(layers, mean_cos_sim, label=phase, color=color)
                        sigma_coef = 1 # ±(sigma_coef σ)の範囲を塗りつぶす
                        plt.fill_between(
                            layers,
                            mean_cos_sim - sigma_coef * std_cos_sim,
                            mean_cos_sim + sigma_coef * std_cos_sim,
                            color=color,
                            alpha=0.1,
                            # label=f"±{sigma_coef} Std Dev"
                        )
                        plt.title(f"Cosine Similarity ({mistype_str})")
                        plt.xlabel("Layer ID")
                        plt.xticks(np.arange(0, len(layers), 1), np.arange(1, len(layers)+1, 1))
                        plt.ylabel("Cosine Similarity")
                        # y軸の範囲を0から1にする
                        plt.ylim(0, 1)
                        plt.legend()
                        plt.grid(True)
                    plt.savefig(save_path, dpi=300, bbox_inches="tight")
                    print(f"Saved {save_path}")
                if metric == "vdiff":
                    # NOTE: (*) こっちの方は個別事例によってニューロンの働きは異なるはずなので，個別ごとのプロットしか無理．
                    #       そのため (num_settings, num_neurons, num_layers) の最初の軸をランダムに選んで，一つの事例に対するプロットを行う
                    mistype_str = "_".join([misclf_type, fpfn]) if fpfn is not None else misclf_type
                    vdiff_save_dir = f"vdiff_plot_{name_vdiff_pick}"
                    os.makedirs(vdiff_save_dir, exist_ok=True)
                    for phase, vals in phase_data.items():
                        # stacked_vdiff = np.stack(vals)
                        stacked_vdiff = vals[seed_for_vdiff]
                        num_layers = stacked_vdiff.shape[-1]
                        for lid in range(num_layers):
                            plt.figure(figsize=(8, 6))
                            save_path = os.path.join(vdiff_save_dir, f"vdiff_{mistype_str}_{phase}_l{lid}.png")
                            interest_arr = stacked_vdiff[:, lid]
                            # 絶対値がtop10%のものを赤くし，それ以外は青にする
                            top10 = np.percentile(np.abs(interest_arr), 90)
                            condition = np.abs(interest_arr).reshape(-1) > top10
                            plt.scatter(np.array(range(len(interest_arr)))[~condition], interest_arr[~condition], alpha=0.1, s=12, color="blue")
                            plt.scatter(np.array(range(len(interest_arr)))[condition], interest_arr[condition], alpha=0.8, s=12, color="red", label="top10 % |Vdiff| neurons")
                            # y=0の線を引く
                            plt.axhline(y=0, color="gray", linestyle="--")
                            plt.xlabel("Neuron id")
                            plt.ylabel("Diff of V-score")
                            plt.legend()
                            plt.savefig(save_path, dpi=300, bbox_inches="tight")
                            print(f"Saved {save_path}")

if __name__ == "__main__":
    # データセットをargparseで受け取る
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str)
    parser.add_argument('k', nargs="?", type=list, help="the fold id (0 to K-1)")
    parser.add_argument('tgt_rank', nargs="?", type=list, help="the rank of the target misclassification type")
    parser.add_argument('n', nargs="?", type=int, help="the factor for the number of neurons to fix")
    parser.add_argument('--misclf_type', type=str, help="the type of misclassification (src_tgt or tgt)", default="tgt")
    parser.add_argument("--fpfn", type=str, help="the type of misclassification (fp or fn)", default=None, choices=["fp", "fn"])
    parser.add_argument("--fl_method", type=str, help="the method used for FL", default="vdiff")
    parser.add_argument("--run_all", action="store_true", help="run all settings")
    args = parser.parse_args()
    ds = args.ds
    k_list = args.k
    tgt_rank_list = args.tgt_rank
    n_list = args.n
    misclf_type = args.misclf_type
    fpfn = args.fpfn
    fl_method = args.fl_method
    run_all = args.run_all

    result_dicts = {}

    if run_all:
        # run_allがtrueなのにkとtgt_rankが指定されている場合はエラー
        assert k_list is None and tgt_rank_list is None and n_list is None, "run_all and k_list or tgt_rank_list or n_list cannot be specified at the same time"
        k_list = range(5)
        tgt_rank_list = range(1, 4)
        n_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 77, 109]
        misclf_type_list = ["all", "src_tgt", "tgt"]
        fpfn_list = [None, "fp", "fn"]
        for k, tgt_rank, n, misclf_type, fpfn in product(k_list, tgt_rank_list, n_list, misclf_type_list, fpfn_list):
            if (misclf_type == "src_tgt" or misclf_type == "all") and fpfn is not None:
                continue
            vdiff_dic = main(ds, k, tgt_rank, n, fl_method, misclf_type, fpfn, run_all=run_all)
            result_dicts[(k, tgt_rank, n, misclf_type, fpfn)] = vdiff_dic
        comb_for_vdiff = list(product(k_list, tgt_rank_list, n_list))
        num_comb = len(comb_for_vdiff)
        seed_for_vdiff = random.randint(0, num_comb - 1)
        k, tgt_rank, n = comb_for_vdiff[seed_for_vdiff]
        name_vdiff_pick = f"k{k}_tgt_rank{tgt_rank}_n{n}"
        aggregate_and_plot(result_dicts, seed_for_vdiff, name_vdiff_pick)
    else:
        assert k_list is not None and tgt_rank_list is not None and n_list is not None, "k_list and tgt_rank_list and n_list should be specified"
        for k, tgt_rank, n in zip(k_list, tgt_rank_list, n_list):
            vdiff_dic = main(ds, k, tgt_rank, n, fl_method, misclf_type, fpfn)
            result_dicts[(k, tgt_rank, n, misclf_type, fpfn)] = vdiff_dic