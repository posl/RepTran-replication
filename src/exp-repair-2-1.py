import os, json, argparse, pickle, time
import numpy as np
import torch
from datasets import load_from_disk
from transformers import DefaultDataCollator, ViTForImageClassification, TrainingArguments, Trainer
from utils.helper import get_device, json2dict
from utils.vit_util import (
    transforms, transforms_c100, identfy_tgt_misclf, get_ori_model_predictions,
    compute_metrics, processor, WeightedTrainer,
    sample_true_positive_indices_per_class, maybe_initialize_repair_weights_
)
from utils.constant import ViTExperiment
from utils.log import set_exp_logging
from logging import getLogger
from peft import LoraConfig, get_peft_model

logger = getLogger("base_logger")

# ────────────────  このスクリプトが使うデフォルト設定 ────────────────
DEFAULT_SETTINGS = {
    "num_sampled_from_correct": 200,   # 正解サンプルからランダム抽出する数
}

TGT_SPLIT  = "repair"
TGT_LAYER  = 11            # 最終層の layernorm 手前をキャッシュしている想定


def count_trainable_params(model, show=True):
    if show:
        print("📋 パラメータ一覧（trainable / untrainable 含む）")

    total_elements = 0
    trainable_elements = 0

    for name, param in model.named_parameters():
        numel = param.numel()
        total_elements += numel
        if param.requires_grad and "modules_to_save" not in name:
            trainable_elements += numel
            status = "✅ trainable"
        else:
            status = "❌ frozen"
        if show:
            print(f"{status:12} | {name:80} | shape: {str(tuple(param.shape)):25} | #params: {numel}")

    if show:
        print("\n📊 Summary")
        print(f"Trainable: {trainable_elements:,} / {total_elements:,} ({100 * trainable_elements / total_elements:.6f}%)")
    return trainable_elements


if __name__ == "__main__":
    # ------------------------------------------------
    # 1. 引数
    # ------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str, choices=["c10", "c100"])
    parser.add_argument("k", type=int)
    parser.add_argument("tgt_rank", type=int)
    parser.add_argument("reps_id", type=int)
    parser.add_argument("--setting_path", type=str)
    parser.add_argument("--misclf_type", type=str, default="tgt")
    parser.add_argument("--include_other_TP_for_fitness", action="store_true")
    parser.add_argument("--fpfn", choices=["fp", "fn"])
    parser.add_argument("--separate_tgt", action="store_true")
    parser.add_argument("--r", type=int, default=1, help="r in LoRA, default=1")
    parser.add_argument("--lora_epoch", type=int, default=20, help="#epochs for LoRA, default=20")
    parser.add_argument("--alpha", type=float, default=0.5, help="alpha for WeightedTrainer (\in [0,1])")
    args = parser.parse_args()
    ds_name = args.ds
    k = args.k
    tgt_rank = args.tgt_rank
    reps_id = args.reps_id
    setting_path = args.setting_path
    misclf_type = args.misclf_type
    fpfn = args.fpfn
    r = args.r
    lora_epoch = args.lora_epoch
    alpha = args.alpha

    # ------------------------------------------------
    # 2. 設定読み込み
    # ------------------------------------------------
    setting_dic = DEFAULT_SETTINGS.copy()
    if args.setting_path:
        setting_dic.update(json2dict(args.setting_path))

    # ------------------------------------------------
    # 3. 基本セットアップ
    # ------------------------------------------------
    device         = get_device()
    pretrained_dir = getattr(ViTExperiment, args.ds).OUTPUT_DIR.format(k=args.k)
    # 結果とかログの保存先を先に作っておく
    # save_dirは, 5種類の誤分類タイプのどれかを一意に表す
    if fpfn is not None and misclf_type == "tgt": # tgt_fp or tgt_fn
        save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_{fpfn}_repair_weight_by_de")
    elif misclf_type == "all": # all
        save_dir = os.path.join(pretrained_dir, f"{misclf_type}_repair_weight_by_de")
    else: # tgt_all or src_tgt
        save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_repair_weight_by_de")
    os.makedirs(save_dir, exist_ok=True)
    lora_save_dir = os.path.join(save_dir, f"exp-repair-2-best_patch_r{r}_alpha_{alpha}_reps{reps_id}")
    metrics_json_path = os.path.join(save_dir, f"exp-repair-2-best_patch_r{r}_alpha_{alpha}_reps{reps_id}.json") # 各設定での実行時間を記録
    # ==================================================================

    tf_func, label_col = (
        (transforms, "label") if args.ds == "c10" else (transforms_c100, "fine_label")
    )
    ds_dirname = f"{args.ds}_fold{args.k}"
    ds         = load_from_disk(os.path.join(ViTExperiment.DATASET_DIR, ds_dirname))
    labels     = {s: np.array(ds[s][label_col]) for s in ["train", "repair", "test"]}
    ds_preprocessed = ds.with_transform(tf_func)

    model, loading_info = ViTForImageClassification.from_pretrained(pretrained_dir, output_loading_info=True)
    model.to(device).eval()
    model = maybe_initialize_repair_weights_(model, loading_info["missing_keys"])
    # このpythonのファイル名を取得
    this_file_name = os.path.basename(__file__).split(".")[0]
    misclf_type_name = misclf_type if fpfn is None else f"{misclf_type}_{fpfn}"
    exp_name = f"{this_file_name}_misclf_top{tgt_rank}_{misclf_type_name}_r{r}_alpha_{alpha}_reps{reps_id}"
    logger = set_exp_logging(exp_dir=pretrained_dir, exp_name=exp_name)
    # argsの情報をログ表示
    logger.info(f"ds_name: {args.ds}, k: {args.k}, tgt_rank: {args.tgt_rank}, reps_id: {args.reps_id}, r: {args.r}, lora_epoch: {args.lora_epoch}, misclf_type: {args.misclf_type}, fpfn: {args.fpfn}, reps_id: {args.reps_id}, alpha: {args.alpha}")

    # ------------------------------------------------
    # 4. ターゲット誤分類サンプルの特定
    # ------------------------------------------------
    misclf_info_dir = os.path.join(pretrained_dir, "misclf_info")
    _, tgt_label, tgt_mis_indices = identfy_tgt_misclf(
        misclf_info_dir,
        tgt_split=TGT_SPLIT,
        tgt_rank=args.tgt_rank,
        misclf_type=args.misclf_type,
        fpfn=args.fpfn,
    )

    indices_to_incorrect = tgt_mis_indices
    pred_res_dir = os.path.join(pretrained_dir, "pred_results", "PredictionOutput")

    # tgtの場合は正解サンプルの定義を(i)全正解サンプルか，(ii)対象のラベルでの正解サンプルに変えることができる
    if args.misclf_type == "tgt":
        ori_pred_labels, _, indices_to_correct_tgt, _, indices_to_correct_others = (
            get_ori_model_predictions(
                pred_res_dir,
                labels,
                tgt_split=TGT_SPLIT,
                misclf_type=args.misclf_type,
                tgt_label=tgt_label,
            )
        )
        # (ii)の対象ラベルでの正解サンプルを選ぶ場合
        if args.separate_tgt:
            indices_to_correct = indices_to_correct_tgt
        # (i)の全部の正解サンプルを選ぶ場合
        else:
            indices_to_correct = np.sort(np.concatenate([indices_to_correct_tgt, indices_to_correct_others]))
    else:
        ori_pred_labels, _, indices_to_correct = get_ori_model_predictions(
            pred_res_dir,
            labels,
            tgt_split=TGT_SPLIT,
            misclf_type=args.misclf_type,
            tgt_label=tgt_label,
        )

    # 正解サンプルは一般に多すぎるのでラベルごとの分布を考慮してサンプリングする
    sampled_indices_to_correct = sample_true_positive_indices_per_class(
        setting_dic["num_sampled_from_correct"],
        indices_to_correct,
        ori_pred_labels,
    )

    tgt_indices = sampled_indices_to_correct.tolist() + indices_to_incorrect.tolist()
    tgt_ds      = ds_preprocessed[TGT_SPLIT].select(tgt_indices)
    tgt_labels  = labels[TGT_SPLIT][tgt_indices]
    # 元々正解/不正解のフラグの列をつける
    # is_correct_flags = np.array([1]*len(sampled_indices_to_correct) + [0]*len(indices_to_incorrect))
    # tgt_ds = tgt_ds.add_column("ori_correct", is_correct_flags.tolist())
    
    # 使用する正解/不正解セットのサイズをチェック
    logger.info(f"len(sampled_indices_to_correct): {len(sampled_indices_to_correct)}, len(indices_to_incorrect): {len(indices_to_incorrect)}, len(tgt_indices): {len(tgt_indices)}")
    (f"len(sampled_indices_to_correct): {len(sampled_indices_to_correct)}, len(indices_to_incorrect): {len(indices_to_incorrect)}, len(tgt_indices): {len(tgt_indices)}")

    # ------------------------------------------------
    # 5. LoRAによるrepair実行
    # ------------------------------------------------
    # LoRAの設定
    config = LoraConfig(
        r=r,
        lora_alpha=r,
        target_modules=["vit.encoder.layer.11.intermediate.repair"],
        lora_dropout=0.1,
        bias="none",
        # modules_to_save=["classifier"],
    )
    lora_model = get_peft_model(model, config)
    num_trainable_params = count_trainable_params(lora_model, show=False) # 訓練できるパラメータ数を表示するときは show=True
    logger.info(f"Number of Trainable parameters: {num_trainable_params:,}")

    # 学習の設定
    batch_size = ViTExperiment.BATCH_SIZE
    training_args = TrainingArguments(
        output_dir=lora_save_dir,
        num_train_epochs=lora_epoch,
        learning_rate=2e-4,
        weight_decay=0.01,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=batch_size,
        remove_unused_columns=False, # img列がないとエラーになるので必要
        evaluation_strategy="no", # エポックの終わりごとにeval_datasetで評価
        logging_strategy="epoch",
        save_strategy="epoch",
        push_to_hub=False,
        disable_tqdm=False,
        log_level="error",
        report_to="tensorboard",
        load_best_model_at_end=False,
    )
    
    # 学習の実行
    data_collator = DefaultDataCollator()
    trainer = WeightedTrainer(
        model=lora_model,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        train_dataset=tgt_ds,
        tokenizer=processor,
        alpha=alpha
    )
    logger.info("Training LoRA model...")
    st = time.perf_counter()
    train_results = trainer.train()
    et = time.perf_counter()
    tot_time = et - st
    logger.info(f"Training completed in {et - st} sec.")
    print(f"Training completed in {et - st} sec.")
    # 実行時間だけをメトリクスとしてjsonに保存
    # (このjsonはあとでrepair rateなども追記される (007f))
    metrics = {"tot_time": tot_time}
    with open(metrics_json_path, "w") as f:
        json.dump(metrics, f)
    
    # ==========================
    # 6. Repairデータセットで推論・保存
    # ==========================
    logger.info(f"Repair dataset size: {len(tgt_ds)} samples")
    logger.info("Predicting on Repair dataset (tgt_ds)...")
    print("Predicting on Repair dataset (tgt_ds)...")
    tgt_ds_pred = trainer.predict(tgt_ds)

    # メトリクスを表示
    logger.info(f"Metrics on repair dataset:")
    for k, v in tgt_ds_pred.metrics.items():
        logger.info(f"  {k}: {v}")

    # pklで保存
    with open(os.path.join(lora_save_dir, "repair_pred.pkl"), "wb") as f:
        pickle.dump(tgt_ds_pred, f)
        
    # ==========================
    # 7. Repair setの予測結果から
    #    repair ratioとbroken ratioを計算
    # ==========================
    # 1. 予測ラベルを取得
    pred_labels = np.argmax(tgt_ds_pred.predictions, axis=1)

    # 2. 対応する正解ラベル
    true_labels = tgt_labels  # これはすでに準備済み
    assert len(pred_labels) == len(true_labels)

    # 3. サンプルごとに「元は正解だったか」「元は誤りだったか」の区別
    # sampled_indices_to_correct, indices_to_incorrect は元のインデックスなので、
    # tgt_indicesにマッピングした順番通りに並んでいる

    # 先にサイズを確認
    n_correct = len(sampled_indices_to_correct) # 元々合ってた
    n_incorrect = len(indices_to_incorrect) # 元々間違ってた

    # サンプルを2グループに分ける
    pred_correct_part = pred_labels[:n_correct]
    true_correct_part = true_labels[:n_correct]
    pred_incorrect_part = pred_labels[n_correct:]
    true_incorrect_part = true_labels[n_correct:]

    # 4. 各グループの精度を計算
    retain_cnt = np.sum(pred_correct_part == true_correct_part)
    repair_cnt = np.sum(pred_incorrect_part == true_incorrect_part)
    retain_ratio = np.mean(pred_correct_part == true_correct_part)  # 元正解サンプルに対して
    repair_ratio = np.mean(pred_incorrect_part == true_incorrect_part)  # 元誤りサンプルに対して

    # 5. 出力
    logger.info(f"retain ratio (元正解サンプル維持率): {retain_ratio:.4f} ({retain_cnt}/{n_correct})")
    logger.info(f"repair ratio (元誤りサンプル修正率): {repair_ratio:.4f} ({repair_cnt}/{n_incorrect})")

    # ==========================
    # 8. Testデータセットで推論・保存
    # ==========================
    test_ds = ds_preprocessed["test"]
    logger.info(f"Test dataset size: {len(test_ds)} samples")
    logger.info("Predicting on Test dataset (test_ds)...")
    print("Predicting on Test dataset (test_ds)...")
    test_ds_pred = trainer.predict(test_ds)

    # メトリクスを表示
    logger.info(f"Metrics on test dataset:")
    for k, v in test_ds_pred.metrics.items():
        logger.info(f"  {k}: {v}")

    # pklで保存
    with open(os.path.join(lora_save_dir, "test_pred.pkl"), "wb") as f:
        pickle.dump(test_ds_pred, f)
        
    # tgt_indicesも保存
    np.save(os.path.join(lora_save_dir, "tgt_indices.npy"), tgt_indices)

    logger.info("All predictions and metrics saved successfully.")