import os, sys, time, pickle, json
import argparse
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import evaluate
import torch
from datasets import load_from_disk
from transformers import ViTForImageClassification
from utils.helper import get_device, json2dict
from utils.vit_util import transforms, transforms_c100, ViTFromLastLayer, identfy_tgt_misclf, get_ori_model_predictions, get_new_model_predictions, get_batched_hs, get_batched_labels, sample_from_correct_samples, sample_true_positive_indices_per_class, maybe_initialize_repair_weights_
from utils.constant import ViTExperiment, Experiment3
from utils.log import set_exp_logging
from utils.de import DE_searcher
from logging import getLogger
from sklearn.metrics import confusion_matrix

logger = getLogger("base_logger")

DEFAULT_SETTINGS = {
    "n": 5, 
    "ratio_sampled_from_correct": 0.05,  # What percentage of correct samples to use for repair
    "num_sampled_from_correct": 200,  # How many samples from correct samples to use for repair
    "max_search_num": 50,
    "pop_size": 100,
    "alpha": 0.5
}

TGT_SPLIT = "repair"
TGT_LAYER = 11

if __name__ == "__main__":
    # Accept dataset via argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str)
    parser.add_argument('k', type=int, help="the fold id (0 to K-1)")
    parser.add_argument('tgt_rank', type=int, help="the rank of the target misclassification type")
    parser.add_argument('reps_id', type=int, help="the repetition id")
    parser.add_argument("--setting_path", type=str, help="path to the setting json file", default=None)
    parser.add_argument("--fl_method", type=str, help="the method used for FL", default="vdiff")
    parser.add_argument('--misclf_type', type=str, help="the type of misclassification (src_tgt or tgt or all)", default="tgt")
    parser.add_argument('--only_eval', action='store_true', help="if True, only evaluate the saved patch", default=False)
    parser.add_argument("--custom_alpha", type=float, help="the custom alpha for the repair", default=None)
    parser.add_argument("--include_other_TP_for_fitness", action="store_true", help="if True, include other TP samples for fitness calculation", default=False)
    parser.add_argument("--fpfn", type=str, help="the type of misclassification (fp or fn)", default=None, choices=["fp", "fn"])
    parser.add_argument("--custom_bounds", type=str, help="the type of bound for the DE search space", default=None, choices=["Arachne", "ContrRep"])
    args = parser.parse_args()
    ds_name = args.ds
    k = args.k
    tgt_rank = args.tgt_rank
    reps_id = args.reps_id
    setting_path = args.setting_path
    fl_method = args.fl_method
    misclf_type = args.misclf_type
    only_eval = args.only_eval
    custom_alpha = args.custom_alpha
    include_other_TP_for_fitness = args.include_other_TP_for_fitness
    fpfn = args.fpfn
    custom_bounds = args.custom_bounds
    print(f"ds_name: {ds_name}, k: {k}, tgt_rank: {tgt_rank}, reps_id: {reps_id}, setting_path: {setting_path}, fl_method: {fl_method}, misclf_type: {misclf_type}, only_eval: {only_eval}, custom_alpha: {custom_alpha}, include_other_TP_for_fitness: {include_other_TP_for_fitness}, fpfn: {fpfn}, custom_bounds: {custom_bounds}")
    
    dataset_dir = ViTExperiment.DATASET_DIR
    exp_obj = getattr(ViTExperiment, ds_name.replace("-", "_"))
    ds = load_from_disk(os.path.join(dataset_dir, f"{ds_name}_fold{k}"))
    pretrained_dir = exp_obj.OUTPUT_DIR.format(k=k)
    if ds_name == "c100":
        tf_func = transforms_c100
        label_col = "fine_label"
    elif ds_name == "tiny-imagenet":
        tf_func = transforms
        label_col = "label"
    else:
        NotImplementedError
    # Get labels (not shuffled)
    labels = {
        "train": np.array(ds["train"][label_col]),
        "repair": np.array(ds["repair"][label_col]),
        "test": np.array(ds["test"][label_col])
    }

    # If a settings JSON file is specified
    if setting_path is not None:
        assert os.path.exists(setting_path), f"{setting_path} does not exist."
        setting_dic = json2dict(setting_path)
        # setting_id becomes the filename in the format setting_{setting_id}.json
        setting_id = os.path.basename(setting_path).split(".")[0].split("_")[-1]
    # If no settings JSON file is specified, use custom or default settings for n and alpha only
    else:
        setting_dic = DEFAULT_SETTINGS
        # If any of custom_alpha, custom_bounds are specified, temporarily set to empty string
        setting_id = "default" if (custom_alpha is None) and (custom_bounds is None) else ""
        parts = []
        if custom_alpha is not None:
            setting_dic["alpha"] = custom_alpha
            parts.append(f"alpha{custom_alpha}")
        if custom_bounds is not None:
            setting_dic["bounds"] = custom_bounds
            parts.append(f"bounds{custom_bounds}")
        # Join list elements with '_'
        setting_id = "_".join(parts)
    # Create save directories for results and logs in advance
    # save_dir uniquely represents one of the 5 types of misclassification
    if fpfn is not None and misclf_type == "tgt": # tgt_fp or tgt_fn
        save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_{fpfn}_repair_weight_by_de")
    elif misclf_type == "all": # all
        save_dir = os.path.join(pretrained_dir, f"{misclf_type}_repair_weight_by_de")
    else: # tgt_all or src_tgt
        save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_repair_weight_by_de")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save file names for repair artifacts
    # ==================================================================
    patch_save_path = os.path.join(save_dir, f"exp-repair-3-1-best_patch_{setting_id}_{fl_method}_reps{reps_id}.npy")
    tracker_save_path = os.path.join(save_dir, f"exp-repair-3-1-tracker_{setting_id}_{fl_method}_reps{reps_id}.pkl")
    # Save tgt_indices of data used for repair as npy
    tgt_indices_save_path = os.path.join(save_dir, f"exp-repair-3-1-tgt_indices_{setting_id}_{fl_method}_reps{reps_id}.npy") # TODO: If randomness is involved in determining tgt_indices, need to make filename that can track it. Even for same setting_id, fl_method but different repetition
    metrics_json_path = os.path.join(save_dir, f"exp-repair-3-1-metrics_for_repair_{setting_id}_{fl_method}_reps{reps_id}.json") # Record execution time for each setting
    # If all the above files already exist, return 0
    if os.path.exists(patch_save_path) and os.path.exists(tracker_save_path) and os.path.exists(tgt_indices_save_path) and os.path.exists(metrics_json_path):
        print(f"All results already exist. Skip this experiment.")
        exit(0)
    # ==================================================================
    
    # Get this Python file name
    this_file_name = os.path.basename(__file__).split(".")[0]
    exp_name = f"{this_file_name}_{setting_id}"
    # Set up logger and display configuration information
    logger = set_exp_logging(exp_dir=save_dir, exp_name=exp_name)
    logger.info(f"ds_name: {ds_name}, fold_id: {k}, setting_path: {setting_path}")
    logger.info(f"setting_dic (id={setting_id}): {setting_dic}")

    tgt_pos = ViTExperiment.CLS_IDX
    # Get device (cuda or cpu)
    device = get_device()
    # Load dataset (takes some time only on first load)
    # Apply preprocessing in real-time when loaded
    ds_preprocessed = ds.with_transform(tf_func)
    # Load pretrained model
    model, loading_info = ViTForImageClassification.from_pretrained(pretrained_dir, output_loading_info=True)
    model.to(device).eval()
    model = maybe_initialize_repair_weights_(model, loading_info["missing_keys"])
    end_li = model.vit.config.num_hidden_layers
    batch_size = ViTExperiment.BATCH_SIZE
    tgt_split = TGT_SPLIT # NOTE: we only use repair split for repairing
    ori_tgt_ds = ds_preprocessed[tgt_split]
    ori_tgt_labels = labels[tgt_split]
    tgt_layer = TGT_LAYER # NOTE: we only use the last layer for repairing
    logger.info(f"tgt_layer: {tgt_layer}, tgt_split: {tgt_split}")

    # ===============================================
    # localization phase
    # ===============================================

    if fpfn is not None and misclf_type == "tgt":
        location_save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_{fpfn}_weights_location")
    elif misclf_type == "all":
        location_save_dir = os.path.join(pretrained_dir, f"all_weights_location")
    else:
        location_save_dir = os.path.join(pretrained_dir, f"misclf_top{tgt_rank}", f"{misclf_type}_weights_location")
    location_filename = "exp-fl-2_location_pareto_weight_bl.npy" # XXX: 今回はこれで固定
    location_save_path = os.path.join(location_save_dir, location_filename)
    pos_before, pos_after = np.load(location_save_path, allow_pickle=True)
    # Display logs
    logger.info(f"pos_before={pos_before}")
    logger.info(f"pos_after={pos_after}")
    logger.info(f"num(pos_to_fix)=num(pos_before)+num(pos_before)={len(pos_before)}+{len(pos_after)}={len(pos_before)+len(pos_after)}")
    print(f"num(pos_to_fix)=num(pos_before)+num(pos_before)={len(pos_before)}+{len(pos_after)}={len(pos_before)+len(pos_after)}")
    

    # ===============================================
    # Data preparation for repair
    # ===============================================

    # Extract misclassification information for tgt_rank
    misclf_info_dir = os.path.join(pretrained_dir, "misclf_info")
    misclf_pair, tgt_label, tgt_mis_indices = identfy_tgt_misclf(misclf_info_dir, tgt_split=tgt_split, tgt_rank=tgt_rank, misclf_type=misclf_type, fpfn=fpfn)
    indices_to_incorrect = tgt_mis_indices
    # NOTE: No need to sample from indices_to_incorrect? Currently using all, so randomness only comes from correct sampling

    # Get correct/incorrect indices for each sample in the repair set of the original model
    pred_res_dir = os.path.join(pretrained_dir, "pred_results", "PredictionOutput")
    if misclf_type == "tgt":
        ori_pred_labels, _, indices_to_correct_tgt, _, indices_to_correct_others = get_ori_model_predictions(pred_res_dir, labels, tgt_split=tgt_split, misclf_type=misclf_type, tgt_label=tgt_label)
        # Select all correct samples even for mis_clf=tgt
        indices_to_correct = np.sort(np.concatenate([indices_to_correct_tgt, indices_to_correct_others]))
    else:
        ori_pred_labels, _, indices_to_correct = get_ori_model_predictions(pred_res_dir, labels, tgt_split=tgt_split, misclf_type=misclf_type, tgt_label=tgt_label)
    print(f"len(indices_to_correct): {len(indices_to_correct)}, len(indices_to_incorrect): {len(indices_to_incorrect)}")
    
    num_sampled_from_pos = int(setting_dic["ratio_sampled_from_correct"] * len(ori_tgt_labels))
    # 正解サンプルは一般に多すぎるのでラベルごとの分布を考慮してサンプリングする
    sampled_indices_to_correct = sample_true_positive_indices_per_class(
        num_sampled_from_pos,   # How many samples from correct samples to use for repair
        indices_to_correct,
        ori_pred_labels,
    )
    ori_sampled_indices_to_correct, ori_indices_to_incorrect = sampled_indices_to_correct.copy(), indices_to_incorrect.copy()
    # Combine extracted correct data and all incorrect data into one dataset
    tgt_indices = sampled_indices_to_correct.tolist() + indices_to_incorrect.tolist() # .tolist() is a non-destructive method
    # ここで (正解サンプル) + (不正解サンプル) の順番にしている
    # Ensure all tgt_indices are unique values
    assert len(tgt_indices) == len(set(tgt_indices)), f"len(tgt_indices): {len(tgt_indices)}, len(set(tgt_indices)): {len(set(tgt_indices))}"
    logger.info(f"tgt_indices: {tgt_indices} (len: {len(tgt_indices)})")
    print(f"len(tgt_indices): {len(tgt_indices)}")
    # Extract data labels corresponding to tgt_indices
    tgt_ds = ori_tgt_ds.select(tgt_indices)
    tgt_labels = ori_tgt_labels[tgt_indices]
    logger.info(f"ori_pred_labels[tgt_indices]: {ori_pred_labels[tgt_indices]} (len: {len(ori_pred_labels[tgt_indices])})")
    logger.info(f"ori_tgt_labels[tgt_indices]: {tgt_labels} (len: {len(tgt_labels)})")
    # print(f"ori_pred_labels[tgt_indices]: {ori_pred_labels[tgt_indices]} (len: {len(ori_pred_labels[tgt_indices])})")
    
    # Get hidden_states_before_layernorm for repair set
    hs_save_dir = os.path.join(pretrained_dir, f"cache_hidden_states_before_layernorm_{tgt_split}")
    hs_save_path = os.path.join(hs_save_dir, f"hidden_states_before_layernorm_{tgt_layer}.npy")
    assert os.path.exists(hs_save_path), f"{hs_save_path} does not exist."
    batch_hs_before_layernorm_tgt = get_batched_hs(hs_save_path, batch_size, tgt_indices, device=device)
    batch_labels_tgt = get_batched_labels(ori_tgt_labels, batch_size, tgt_indices)
    # Also create batch for entire repair set (without specifying tgt_indices)
    batch_hs_before_layernorm = get_batched_hs(hs_save_path, batch_size, device=device)
    batch_labels = get_batched_labels(ori_tgt_labels, batch_size)

    # NOTE: Random seed is not considered even for the same setting_id
    np.save(tgt_indices_save_path, tgt_indices)

    # ===============================================
    # DE search (patch generation)
    # ===============================================

    # Prepare model with only the final layer
    vit_from_last_layer = ViTFromLastLayer(model)
    vit_from_last_layer.eval()

    # Check prediction results for entire repair set and repair target data on pretrained ViT immediately after loading
    # entire repair set
    ori_pred_labels, ori_true_labels = get_new_model_predictions(vit_from_last_layer, batch_hs_before_layernorm, batch_labels, tgt_pos=tgt_pos)
    ori_is_correct = ori_pred_labels == ori_true_labels
    # repair target data
    ori_pred_labels_tgt, ori_true_labels_tgt = get_new_model_predictions(vit_from_last_layer, batch_hs_before_layernorm_tgt, batch_labels_tgt, tgt_pos=tgt_pos)
    ori_is_correct_tgt = ori_pred_labels_tgt == ori_true_labels_tgt
    # Display logs
    logger.info(f"sum(ori_is_correct), len(ori_is_correct), {sum(ori_is_correct), len(ori_is_correct)}")
    logger.info(f"sum(ori_is_correct_tgt), len(ori_is_correct_tgt), {sum(ori_is_correct_tgt), len(ori_is_correct_tgt)}")

    # Make weights at pos_before, pos_after positions optimization variables
    linear_before2med = vit_from_last_layer.base_model_last_layer.intermediate.dense # on GPU
    weight_before2med = linear_before2med.weight.cpu().detach().numpy() # on CPU
    linear_med2after = vit_from_last_layer.base_model_last_layer.output.dense # on GPU
    weight_med2after = linear_med2after.weight.cpu().detach().numpy() # on CPU

    # Initialize DE_searcher
    max_search_num = setting_dic["max_search_num"]
    alpha = setting_dic["alpha"] # [0, 1] value where weight for correct classification in fitness: weight for misclassification = 1-alpha: alpha
    assert 0 <= alpha <= 1, f"alpha should be in [0, 1]. alpha: {alpha}"
    logger.info(f"alpha of the fitness function: {alpha}")
    pop_size = setting_dic["pop_size"]
    num_labels = len(set(labels["train"]))
    # Update correct, incorrect indices
    indices_to_correct_for_de_searcher = np.arange(len(sampled_indices_to_correct))
    indices_to_incorrect_for_de_searcher = np.arange(len(sampled_indices_to_correct), len(sampled_indices_to_correct) + len(indices_to_incorrect))
    logger.info(f"indices_to_correct_for_de_searcher: {indices_to_correct_for_de_searcher}, indices_to_incorrect_for_de_searcher: {indices_to_incorrect_for_de_searcher}")
    # Confirm that the model is correct for the first indices_to_correct_for_de_searcher samples of batch_hs_before_layernorm_tgt and incorrect for the remaining indices_to_incorrect_for_de_searcher samples
    assert np.sum(ori_is_correct_tgt[indices_to_correct_for_de_searcher]) == len(indices_to_correct_for_de_searcher), f"ori_is_correct_tgt[indices_to_correct_for_de_searcher]: {ori_is_correct_tgt[indices_to_correct_for_de_searcher]}"
    assert np.sum(ori_is_correct_tgt[indices_to_incorrect_for_de_searcher]) == 0, f"ori_is_correct_tgt[indices_to_incorrect_for_de_searcher]: {ori_is_correct_tgt[indices_to_incorrect_for_de_searcher]}"
    
    # logger.info("Before DE search...")
    # Initialize DE_searcher
    searcher = DE_searcher(
        batch_hs_before_layernorm=batch_hs_before_layernorm_tgt,
        batch_labels=batch_labels_tgt,
        indices_to_correct=indices_to_correct_for_de_searcher,
        indices_to_wrong=indices_to_incorrect_for_de_searcher,
        num_label=num_labels,
        indices_to_target_layers=[tgt_layer],
        device=device,
        mutation=(0.5, 1),
        recombination=0.7,
        max_search_num=max_search_num,
        partial_model=vit_from_last_layer,
        alpha=alpha,
        pop_size=pop_size,
        mode="weight",
        pos_before=pos_before,
        pos_after=pos_after,
        weight_before2med=weight_before2med,
        weight_med2after=weight_med2after,
        custom_bounds=custom_bounds
    )
    logger.info(f"Start DE search...")
    s = time.perf_counter()
    patch, fitness_tracker = searcher.search(patch_save_path=patch_save_path, pos_before=pos_before, pos_after=pos_after, tracker_save_path=tracker_save_path)
    e = time.perf_counter()
    # print(f"[after DE] {vit_from_last_layer.base_model_last_layer.intermediate.dense.weight.data[pos_before[0][0]][pos_before[0][1]]}")
    tot_time = e - s
    logger.info(f"Total execution time: {tot_time} sec.")
    print(f"Total execution time: {tot_time} sec.")
    # Save only execution time as metrics to json
    # (This json will later have repair rate etc. added (007f))
    metrics = {"tot_time": tot_time}
    with open(metrics_json_path, "w") as f:
        json.dump(metrics, f)