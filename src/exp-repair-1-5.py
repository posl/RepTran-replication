import os, re, json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
from utils.constant import Experiment3, ExperimentRepair1, ExperimentRepair2, ViTExperiment

NUM_REPS = 1
DEFAULT_SETTINGS = {
    "n": 5,
    "num_sampled_from_correct": 200,
    "max_search_num": 50,
    "pop_size": 100,
    "alpha": 0.5
}

def parse_filename(filename):
    m = re.match(r"exp-repair-1-metrics_for_(?:repair|test)_(.+)_(.+)_reps(\d+)\.json", filename)
    if m:
        return m.group(1), m.group(2), int(m.group(3))  # setting_id, fl_method, reps_id
    else:
        return None, None, None

def parse_setting_id(setting_id):
    wnum_val, n_val, alpha_val = None, None, None
    w = re.search(r"wnum(\d+)", setting_id)
    if w:
        wnum_val = int(w.group(1))

    n_ = re.search(r"n([\d\.]+)", setting_id)
    if n_:
        maybe_n = float(n_.group(1))
        n_val = int(maybe_n) if maybe_n.is_integer() else maybe_n

    a_ = re.search(r"alpha([\d\.]+)", setting_id)
    if a_:
        alpha_val = float(a_.group(1))
    return n_val, wnum_val, alpha_val

def process_one_json(json_path, ds_name="c100", tgt_rank=None, misclf_type=None, fpfn=None):
    filename = os.path.basename(json_path)
    setting_id, fl_method, reps_id = parse_filename(filename)
    if setting_id is None:
        return None

    # print(json_path)
    with open(json_path, "r") as f:
        data = json.load(f)

    n_val, wnum_val, alpha_val = parse_setting_id(setting_id)

    row = {}
    row["subject"] = "ViT/C100"
    row["method"] = fl_method if fl_method else "(unknown)"
    row["tgt_rank"] = tgt_rank
    row["tgt. misclf. type"] = misclf_type if fpfn is None else "_".join([misclf_type, fpfn])

    row["alpha"] = alpha_val
    if wnum_val is not None:
        row["#modified weights"] = wnum_val
    else:
        row["#modified weights"] = 8 * n_val * n_val if n_val is not None else "(N/A)"

    row["RR_tgt"] = data.get("repair_rate_tgt", None)
    row["BR_all"] = data.get("break_rate_overall", None)
    row["ΔACC."]  = data.get("delta_acc", None)
    row["TOT_TIME"] = data.get("tot_time", None)
    return row

if __name__ == "__main__":
    ds = "c100"
    k_list = [0]
    tgt_rank_list = range(1, 4) # TODO: SHOULD BE range(1, 6)
    misclf_type_list = ["src_tgt", "tgt"]
    fpfn_list = [None, "fp", "fn"]
    alpha_list = [0.2, 0.4, 0.6, 0.8]
    fl_method_list = ["vmg", "bl", "random"]
    exp_list = [ExperimentRepair1, ExperimentRepair2]
    tgt_split_list = ["repair", "test"]

    for tgt_split in tgt_split_list:
        big_records = []
        for k in k_list:
            pretrained_dir = getattr(ViTExperiment, ds).OUTPUT_DIR.format(k=k)
            for tgt_rank in tgt_rank_list:
                for misclf_type in misclf_type_list:
                    for fpfn in fpfn_list:
                        for fl_method in fl_method_list:
                            for alpha in alpha_list:
                                for exp in exp_list:
                                    if (misclf_type == "src_tgt") and (fpfn is not None):
                                        continue
                                    if misclf_type == "all" and tgt_rank != 1:
                                        continue
                                    if fl_method == "vmg":
                                        n = exp.NUM_IDENTIFIED_NEURONS_RATIO
                                        wnum = exp.NUM_IDENTIFIED_WEIGHTS
                                        wnum = 8 * wnum * wnum
                                    else:
                                        n = exp.NUM_IDENTIFIED_WEIGHTS
                                        wnum = None
                                    if fpfn is not None and misclf_type == "tgt":
                                        save_dir = os.path.join(
                                            pretrained_dir,
                                            f"misclf_top{tgt_rank}",
                                            f"{misclf_type}_{fpfn}_repair_weight_by_de"
                                        )
                                    elif misclf_type == "all":
                                        save_dir = os.path.join(pretrained_dir,
                                            f"{misclf_type}_repair_weight_by_de")
                                    else:
                                        save_dir = os.path.join(
                                            pretrained_dir,
                                            f"misclf_top{tgt_rank}",
                                            f"{misclf_type}_repair_weight_by_de"
                                        )
                                    # setting_id
                                    parts = []
                                    if n is not None:
                                        if n >= 1:
                                            n = int(n)
                                        parts.append(f"n{n}")
                                    if wnum is not None:
                                        parts.append(f"wnum{wnum}")
                                    if alpha is not None:
                                        parts.append(f"alpha{alpha}")
                                    parts.append("boundsArachne")
                                    setting_id = "_".join(parts)

                                    for reps_id in range(NUM_REPS):
                                        metrics_json_path = os.path.join(
                                            save_dir,
                                            f"exp-repair-1-metrics_for_{tgt_split}_{setting_id}_{fl_method}_reps{reps_id}.json"
                                        )
                                        if not os.path.exists(metrics_json_path):
                                            # エラー終了
                                            print(f"[ERROR] Not found: {metrics_json_path}")
                                            raise FileNotFoundError(metrics_json_path)
                                        # print(f"[INFO] Processing: {metrics_json_path}")
                                        row_data = process_one_json(
                                            json_path=metrics_json_path,
                                            ds_name=ds,
                                            tgt_rank=tgt_rank,
                                            misclf_type=misclf_type,
                                            fpfn=fpfn,
                                        )
                                        # print(f"row_data: {row_data}")
                                        if row_data:
                                            big_records.append(row_data)

        # ---------------------------
        # (A) でかいCSVを保存 (alpha と tgt_rank 列を含む)
        # ---------------------------
        df_all = pd.DataFrame(big_records, columns=[
            "subject",
            "method",
            "tgt. misclf. type",
            "tgt_rank",
            "#modified weights",
            "alpha",
            "RR_tgt",
            "BR_all",
            "ΔACC.",
            "TOT_TIME",
        ])
        print(df_all.shape)
        df_all.to_csv(f"exp-repair-1-5_{tgt_split}_results_all.csv", index=False)
        print(f"[INFO] Saved big CSV => exp-repair-1-5_{tgt_split}_results_all.csv")

        # 数値列を float に変換
        for col in ["RR_tgt", "BR_all", "ΔACC.", "TOT_TIME"]:
            df_all[col] = pd.to_numeric(df_all[col], errors='coerce')

        # ---------------------------
        # (B) alpha でグルーピングして平均 → alpha列はいらない
        #     例: (method, tgt. misclf. type, tgt_rank, #modified weights) ごとに平均
        # ---------------------------
        if tgt_split == "repair":
            df_grouped = (
                df_all.groupby(["method", "tgt. misclf. type", "#modified weights"], as_index=False)
                .agg({
                    "RR_tgt": "mean",
                    "BR_all": "mean",
                    "ΔACC.": "mean",
                    "TOT_TIME": "mean"
                })
            )
        else:
            df_grouped = (
                df_all.groupby(["method", "tgt. misclf. type", "#modified weights"], as_index=False)
                .agg({
                    "RR_tgt": "mean",
                    "BR_all": "mean",
                    "ΔACC.": "mean"
                    # テストセット列があればそれらも同様に加えてOK
                })
            )

        # これで alpha はグループ化に含めず、そのぶん列にも残りません(平均後のテーブルから除外)

        df_grouped.to_csv(f"exp-repair-1-5_{tgt_split}_results_averaged.csv", index=False)
        print(f"[INFO] Saved averaged CSV => exp-repair-1-5_{tgt_split}_results_averaged.csv")


        # 1) 可視化用の設定を追加
        rename_map = {"vmg": "Ours", "random": "Random", "bl": "Arachne"}
        method_order = ["Ours", "Random", "Arachne"]  # x 軸や hue の並び順
        my_palette = {
            "Ours":    "C0",  # Seaborn/matplotlibの色指定 (C0, C1, C2 などで揃える)
            "Random":  "C1",
            "Arachne": "C2"
        }

        metrics = ["RR_tgt", "BR_all", "ΔACC.", "TOT_TIME"] if tgt_split == "repair" else ["RR_tgt", "BR_all", "ΔACC."]
        sub_misclf_types = ["src_tgt", "tgt_fp", "tgt_fn", "tgt"]

        # 2) 再度 float 化（念のため）
        df_all["RR_tgt"] = pd.to_numeric(df_all["RR_tgt"], errors="coerce")
        df_all["BR_all"] = pd.to_numeric(df_all["BR_all"], errors="coerce")
        df_all["ΔACC."]  = pd.to_numeric(df_all["ΔACC."],  errors="coerce")
        if tgt_split == "repair":
            df_all["TOT_TIME"]  = pd.to_numeric(df_all["TOT_TIME"],  errors="coerce")

        # =========================
        # (C) バイオリン図 (4×3)
        # =========================
        fig, axes = plt.subplots(nrows=len(sub_misclf_types), ncols=len(metrics), figsize=(18, 12))

        for i, mt in enumerate(sub_misclf_types):
            for j, metric in enumerate(metrics):
                ax = axes[i][j]

                subset = df_all[df_all["tgt. misclf. type"] == mt].copy()
                if len(subset) == 0:
                    ax.set_visible(False)
                    continue
                
                # ★ 追加: メソッド名を置き換え
                subset["method"] = subset["method"].map(rename_map)

                # バイオリンプロット (method_order & my_palette を明示)
                sns.violinplot(
                    data=subset,
                    x="method",
                    y=metric,
                    cut=0,
                    scale="width",
                    inner="box",
                    order=method_order,
                    palette=my_palette,
                    ax=ax
                )
                ax.set_title(f"{mt} - {metric}", fontsize=12)
                ax.set_xlabel("Method", fontsize=10)
                ax.set_ylabel(metric, fontsize=10)

        plt.tight_layout()
        plt.savefig(f"exp-repair-1-5_{tgt_split}_violin.png", dpi=150)
        print(f"[INFO] Saved violin plot => exp-repair-1-5_{tgt_split}_violin.png")


        # =========================
        # (D) alphaごとの遷移を示す折れ線グラフ (4×3)
        # =========================
        fig2, axes2 = plt.subplots(nrows=len(sub_misclf_types), ncols=len(metrics), figsize=(18, 16))

        for i, mt in enumerate(sub_misclf_types):
            for j, metric in enumerate(metrics):
                ax2 = axes2[i][j]

                subset_line = df_all[df_all["tgt. misclf. type"] == mt].copy()
                subset_line = subset_line[subset_line["alpha"].notnull()]
                if len(subset_line) == 0:
                    ax2.set_visible(False)
                    continue

                # ★ メソッド名を置き換え
                subset_line["method"] = subset_line["method"].map(rename_map)

                subset_line.sort_values("alpha", inplace=True)

                sns.lineplot(
                    data=subset_line,
                    x="alpha",
                    y=metric,
                    hue="method",
                    hue_order=method_order,
                    palette=my_palette,
                    marker="o",
                    dashes=False,
                    ax=ax2  
                )

                ax2.set_title(f"{mt} - {metric}", fontsize=12)
                ax2.set_xlabel("alpha", fontsize=10)
                ax2.set_ylabel(metric, fontsize=10)
                ax2.set_xticks([0.2, 0.4, 0.6, 0.8])
                ax2.set_xlim(0.2, 0.8)

        plt.tight_layout()
        plt.savefig(f"exp-repair-1-5_{tgt_split}_alpha_lineplots.png", dpi=150)
        print(f"[INFO] Saved alpha transition line plots => exp-repair-1-5_{tgt_split}_alpha_lineplots.png")