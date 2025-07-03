import os, re, json, sys, math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ライブラリ側の定数クラス
from utils.constant import ViTExperiment

NUM_REPS = 5
alpha_in_arachne_list = [1, 4, 8, 10]                       # ← ご指定
alpha_ratio_list      = [a / (1 + a) for a in alpha_in_arachne_list]

def collect_records(ds: str, tgt_split: str) -> pd.DataFrame:
    """
    指定データセット（c100 / tiny-imagenet）と split（repair / test）に対し
    JSON 群を走査してレコード DataFrame を返す
    """
    big = []
    k_list          = [0]
    tgt_rank_list   = [1, 2, 3]
    misclf_list = ["src_tgt", "tgt_fp", "tgt_fn"]
    fl_method_li    = ["ours", "bl", "random"]
    w_num_list = [236] # exp-repair-5.md 参照

    for k in k_list:
        pretrained_dir = getattr(ViTExperiment, ds.replace("-", "_")).OUTPUT_DIR.format(k=k)
        for tgt_rank in tgt_rank_list:
            for misclf_full in misclf_list:
                save_dir = os.path.join(
                    pretrained_dir,
                    f"misclf_top{tgt_rank}",
                    f"{misclf_full}_repair_weight_by_de"   # ← これだけで OK
                )
                for flm in fl_method_li:
                    for alpha_raw, alpha_ratio in zip(alpha_in_arachne_list, alpha_ratio_list):
                        exp_id = 4 if alpha_raw == 10 else 5
                        for w_num in w_num_list:
                            # --- setting_id（※ JSON ファイル名と一致） ----------------------------
                            parts = []
                            # n か wnum が無いと JSON が見つからないことがあるので元スクリプトと同じ生成規則
                            parts.append(f"n{int(w_num) if w_num >= 1 else w_num}")
                            parts.append(f"alpha{alpha_ratio}")
                            parts.append("boundsArachne")
                            setting_id = "_".join(parts)

                            # --- reps ループ -----------------------------------------------------
                            for reps in range(NUM_REPS):
                                jp = os.path.join(
                                    save_dir,
                                    f"exp-repair-{exp_id}-1-metrics_for_{tgt_split}_{setting_id}_{flm}_reps{reps}.json",
                                )
                                if not os.path.exists(jp):
                                    raise FileNotFoundError(jp)

                                with open(jp) as f:
                                    js = json.load(f)

                                big.append(
                                    {
                                        "dataset":   ds,
                                        "tgt. misclf. type": misclf_full,  # src_tgt / tgt_fp / tgt_fn
                                        "tgt_rank": tgt_rank,  # 1 / 2 / 3
                                        "method":    flm,
                                        "#modified weights": w_num,  # 236
                                        "alpha_raw": alpha_raw,   # 1 / 4 / 8 / 10
                                        "alpha":     alpha_ratio, # 0.5 / 0.8 / … / 0.909…
                                        "reps_id": reps,
                                        "RR_tgt":    js.get("repair_rate_tgt"),
                                        "BR_all":    js.get("break_rate_overall"),
                                        "ΔACC.":     js.get("delta_acc"),
                                        "TOT_TIME": js.get("tot_time"),
                                    }
                                )
    return pd.DataFrame(big)


def plot_alpha(df: pd.DataFrame, ds: str, tgt_split: str):
    """
    α_line plot を描画して png 保存
    """
    replace = {"ours": "Ours", "bl": "Arachne", "random": "Random"}
    df["method"] = df["method"].map(replace)

    # α(axis) をカテゴリとして登録 → 幅を均等に
    alpha_cats = [str(a) for a in alpha_in_arachne_list]  # '1','4','8','10'
    df["α_cat"] = pd.Categorical(df["alpha_raw"].astype(str), categories=alpha_cats, ordered=True)

    hue_order = ["Ours", "Arachne", "Random"]
    palette   = {"Ours": "C0", "Arachne": "C2", "Random": "C1"}
    metrics   = ["RR_tgt", "BR_all", "ΔACC."]
    row_types = ["src_tgt", "tgt_fp", "tgt_fn", "tgt"]

    fig, axes = plt.subplots(len(row_types), len(metrics), figsize=(18, 16))
    for i, mt in enumerate(row_types):
        subset = df[df["tgt. misclf. type"] == mt]
        for j, metric in enumerate(metrics):
            ax = axes[i][j]
            if subset.empty:
                ax.axis("off")
                continue
            sns.lineplot(
                data=subset,
                x="α_cat",
                y=metric,
                hue="method",
                hue_order=hue_order,
                palette=palette,
                marker="o",
                dashes=False,
                sort=False,  # α_cat はカテゴリなので False で順序固定
                ax=ax,
            )
            ax.set_title(f"{mt} – {metric}", fontsize=12)
            ax.set_xlabel("alpha (Arachne parameter)")
    plt.tight_layout()
    out_png = f"exp-repair-5-1_{ds}_{tgt_split}_alpha_lineplots.png"
    plt.savefig(out_png, dpi=150)
    print("[INFO] saved", out_png)


# ╭───────────────── エントリーポイント ─────────────────╮
if __name__ == "__main__":
    ds_list = ["c100", "tiny-imagenet"]
    tgt_split_list = ["repair", "test"]
    
    for ds_arg in ds_list:
        for tgt_split_arg in tgt_split_list:
            print(f"{'='*90}\nProcessing: dataset={ds_arg}, split={tgt_split_arg}")
            # --- データ収集 & 保存 --------------------------------------------------
            df_all = collect_records(ds_arg, tgt_split_arg)
            out_csv = f"exp-repair-5-1_{ds_arg}_{tgt_split_arg}_results_all.csv"
            df_all.to_csv(out_csv, index=False)
            print("[INFO] saved", out_csv)

            # --- プロット ----------------------------------------------------------
            plot_alpha(df_all, ds_arg, tgt_split_arg)
            
            # ========= α と RR / BR 相関 (dataset × misclf_type × method) =========
            from scipy.stats import pearsonr

            def corr_by_method(df: pd.DataFrame, metric: str) -> pd.DataFrame:
                """metric = 'RR_tgt' or 'BR_all'"""
                res = (
                    df
                    .groupby(["dataset", "tgt. misclf. type", "method"], dropna=True)
                    .apply(lambda g: pearsonr(g["alpha_raw"], g[metric])[0] if len(g) > 1 else float("nan"))
                    .reset_index(name=f"corr_{metric}")
                )
                return res

            if tgt_split_arg == "test":
                # ① α–RR 相関
                corr_rr = corr_by_method(df_all, "RR_tgt")
                # ② α–BR 相関
                corr_br = corr_by_method(df_all, "BR_all")

                # ── ピボットして横持ち化 ──
                tab_rr = (
                    corr_rr.pivot(index=["dataset", "tgt. misclf. type"],
                                columns="method", values="corr_RR_tgt")
                        .rename(columns={"vmg":"Ours", "bl":"Arachne", "random":"Random"})
                )

                tab_br = (
                    corr_br.pivot(index=["dataset", "tgt. misclf. type"],
                                columns="method", values="corr_BR_all")
                        .rename(columns={"vmg":"Ours", "bl":"Arachne", "random":"Random"})
                )

                # ── RR と BR を左右に結合し、列順を整える ──
                table = pd.concat([tab_rr, tab_br], axis=1, keys=["RR", "BR"])
                table = table.reindex(columns=pd.MultiIndex.from_product(
                            [["RR","BR"], ["Ours","Arachne","Random"]]))

                # ── CSV 保存 & コンソール表示 ──
                out_corr = f"exp-repair-5-1_{ds_arg}_{tgt_split_arg}_corr_alpha_full.csv"
                table.to_csv(out_corr)
                print("\n=== α vs RR / BR correlation table ===")
                print(table)
                print("[INFO] saved", out_corr)
