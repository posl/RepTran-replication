import os, sys, time, pickle, json, math
import argparse
import torch
import copy
from tqdm import tqdm
from collections import defaultdict
from itertools import product
import numpy as np
from utils.helper import get_device
from utils.vit_util import transforms, transforms_c100, ViTFromLastLayer
from utils.constant import ViTExperiment
from utils.de import set_new_weights, check_new_weights
from datasets import load_from_disk
from transformers import ViTForImageClassification


def get_save_dir(pretrained_dir, tgt_rank, misclf_type, fpfn):
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
    return save_dir


def main(ds_name, k, tgt_rank, n, fl_method, misclf_type, fpfn, run_all):
    device = get_device()
    print(
        f"ds_name: {ds_name}, fold_id: {k}, tgt_rank: {tgt_rank}, n: {n}, fl_method: {fl_method}, misclf_type: {misclf_type}, fpfn: {fpfn}"
    )
    # Pretrained model directory
    pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR.format(k=k)

    # Create save destination for results and logs in advance
    save_dir = get_save_dir(pretrained_dir, tgt_rank, misclf_type, fpfn)
    os.makedirs(save_dir, exist_ok=True)

    # Check if fl_method is a list
    if isinstance(fl_method, list):
        # If fl_method is a list, change save_dir
        location_save_path = {}
        pos_before_after = {}
        for fm in fl_method:
            location_save_path[fm] = os.path.join(save_dir, f"location_n{n}_{fm}.npy")
            # Extract localized position
            pos_before_after[fm] = np.load(location_save_path[fm])
        proba_save_dir = os.path.join(save_dir, f"proba_n{n}_{'_'.join(fl_method)}")
        os.makedirs(proba_save_dir, exist_ok=True)
        pos_before, pos_after = None, None
    else:
        location_save_path = os.path.join(save_dir, f"location_n{n}_{fl_method}.npy")
        # Extract localized position
        pos_before, pos_after = np.load(location_save_path)
        proba_save_dir = os.path.join(save_dir, f"proba_n{n}_{fl_method}")
        os.makedirs(proba_save_dir, exist_ok=True)
    print(
        f"save_dir: {save_dir},\n location_save_path: {location_save_path},\n proba_save_dir: {proba_save_dir}"
    )

    # Load dataset (takes some time only on first load)
    ds_dirname = f"{ds_name}_fold{k}"
    ds = load_from_disk(os.path.join(ViTExperiment.DATASET_DIR, ds_dirname))
    tgt_split = "repair"
    tgt_layer = 11
    # Different variable sets for each dataset
    if ds_name == "c10" or ds_name == "c10c":
        tf_func = transforms
        label_col = "label"
    elif ds_name == "c100" or ds_name == "c100c":
        tf_func = transforms_c100
        label_col = "fine_label"
    else:
        NotImplementedError
    true_labels = ds[tgt_split][label_col]

    # Directory for cache storage
    cache_dir = os.path.join(
        pretrained_dir, f"cache_hidden_states_before_layernorm_{tgt_split}"
    )
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(
        cache_dir, f"hidden_states_before_layernorm_{tgt_layer}.npy"
    )
    # Confirm that cache_path exists
    assert os.path.exists(cache_path), f"cache_path: {cache_path} does not exist."
    # Check the cached hidden states and ViTFromLastLayer
    cached_hidden_states = np.load(cache_path)
    hidden_states_before_layernorm = torch.from_numpy(cached_hidden_states).to(device)
    batch_size = ViTExperiment.BATCH_SIZE
    num_batches = (
        hidden_states_before_layernorm.shape[0] + batch_size - 1
    ) // batch_size  # Calculate number of batches (round up because we want to use the last incomplete batch)
    batched_hidden_states_before_layernorm = np.array_split(
        hidden_states_before_layernorm, num_batches
    )

    if isinstance(fl_method, list):
        if "vdiff_asc" in fl_method:
            pos_asc = pos_before_asc, pos_after_asc = pos_before_after["vdiff_asc"]
            print(
                f"pos_before_asc: {pos_before_asc.shape}, pos_after_asc: {pos_after_asc.shape}"
            )
        if "vdiff_desc" in fl_method:
            pos_desc = pos_before_desc, pos_after_desc = pos_before_after["vdiff_desc"]
            print(
                f"pos_before_desc: {pos_before_desc.shape}, pos_after_desc: {pos_after_desc.shape}"
            )

        if "vdiff_asc" in fl_method and "vdiff_desc" in fl_method:
            # Check if there are overlaps between pos_before_asc and desc, and pos_after_asc and desc respectively
            for l, pos in zip(
                ["intermediate", "output"],
                [(pos_before_asc, pos_before_desc), (pos_after_asc, pos_after_desc)],
            ):
                # Convert index list to set of 2D tuples
                set1 = {tuple(pair) for pair in pos[0]}
                set2 = {tuple(pair) for pair in pos[1]}
                assert len(set1) == len(pos[0]), f"pos_before_{l} has duplicates"
                assert len(set2) == len(pos[1]), f"pos_after_{l} has duplicates"
                duplicates = set1.intersection(set2)
                # If there are duplicates, terminate with assertion violation
                assert (
                    len(duplicates) == 0
                ), f"pos_before_{l} and pos_after_{l} have duplicates"
            # Set pos_*_asc to x0 and pos_*_desc to x2, or vice versa
            op_dict = {
                "as_de": {"asc": "sup", "desc": "enh"},
                "ae_ds": {"asc": "enh", "desc": "sup"},
            }
            rank_list = ["asc", "desc"]
            pos_list = [pos_asc, pos_desc]
        elif "vdiff_desc" in fl_method:
            op_dict = {"de": {"desc": "enh"}, "ds": {"desc": "sup"}}
            rank_list = ["desc"]
            pos_list = [pos_desc]
        elif "vdiff_asc" in fl_method:
            op_dict = {"ae": {"asc": "enh"}, "as": {"asc": "sup"}}
            rank_list = ["asc"]
            pos_list = [pos_asc]
        else:
            raise ValueError(f"Invalid fl_method: {fl_method}")
        for op_id, op in op_dict.items():
            # Load trained model
            model = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
            model.eval()
            # Create ViTFromLastLayer instance
            vit_from_last_layer = copy.deepcopy(ViTFromLastLayer(model))
            for rank, pos in zip(rank_list, pos_list):
                pos_before, pos_after = pos
                # print(f"pos_before: {pos_before.shape}, pos_after: {pos_after.shape}")
                dummy_in = [0] * (len(pos_before) + len(pos_after))
                set_new_weights(
                    dummy_in, pos_before, pos_after, vit_from_last_layer, op=op[rank]
                )
            # TODO: At this point, the model weights have been set. Here we want to validate whether the weights have been set as expected
            for rank, pos in zip(rank_list, pos_list):
                print(f"Checking the weights for {rank}...")
                pos_before, pos_after = pos
                temp_model = ViTFromLastLayer(model)
                # Check if weights have been changed
                check_new_weights(
                    dummy_in,
                    pos_before,
                    pos_after,
                    temp_model,
                    vit_from_last_layer,
                    op=op[rank],
                )

            all_logits = []
            all_proba = []
            for cached_state in tqdm(
                batched_hidden_states_before_layernorm,
                total=len(batched_hidden_states_before_layernorm),
            ):
                logits = vit_from_last_layer(
                    hidden_states_before_layernorm=cached_state
                )
                proba = torch.nn.functional.softmax(logits, dim=-1)
                logits = logits.detach().cpu().numpy()
                proba = proba.detach().cpu().numpy()
                all_logits.append(logits)
                all_proba.append(proba)
            all_logits = np.concatenate(all_logits, axis=0)
            all_proba = np.concatenate(all_proba, axis=0)
            all_pred_labels = all_logits.argmax(axis=-1)

            # Extract proba for each value of true_pred_labels
            proba_dict = defaultdict(list)
            for true_label, proba in zip(true_labels, all_proba):
                proba_dict[true_label].append(proba)
            for true_label, proba_list in proba_dict.items():
                proba_dict[true_label] = np.stack(proba_list)
            for true_label, proba in proba_dict.items():
                save_path = os.path.join(
                    proba_save_dir, f"{tgt_split}_proba_{op_id}_{true_label}.npy"
                )
                np.save(save_path, proba)
    else:
        for op in ["enh", "sup"]:
            # Load trained model
            model = ViTForImageClassification.from_pretrained(pretrained_dir).to(device)
            model.eval()
            # Create ViTFromLastLayer instance
            vit_from_last_layer = ViTFromLastLayer(model)
            if op is not None:
                # Change model weights
                dummy_in = [0] * (len(pos_before) + len(pos_after))
                set_new_weights(
                    dummy_in, pos_before, pos_after, vit_from_last_layer, op=op
                )
            all_logits = []
            all_proba = []
            for cached_state in tqdm(
                batched_hidden_states_before_layernorm,
                total=len(batched_hidden_states_before_layernorm),
            ):
                logits = vit_from_last_layer(
                    hidden_states_before_layernorm=cached_state
                )
                proba = torch.nn.functional.softmax(logits, dim=-1)
                logits = logits.detach().cpu().numpy()
                proba = proba.detach().cpu().numpy()
                all_logits.append(logits)
                all_proba.append(proba)
            all_logits = np.concatenate(all_logits, axis=0)
            all_proba = np.concatenate(all_proba, axis=0)
            all_pred_labels = all_logits.argmax(axis=-1)
            is_correct = np.equal(all_pred_labels, true_labels)

            # Extract proba for each value of true_pred_labels
            proba_dict = defaultdict(list)
            for true_label, proba in zip(true_labels, all_proba):
                proba_dict[true_label].append(proba)
            for true_label, proba_list in proba_dict.items():
                proba_dict[true_label] = np.stack(proba_list)
            for true_label, proba in proba_dict.items():
                save_path = os.path.join(
                    proba_save_dir, f"{tgt_split}_proba_{op}_{true_label}.npy"
                )
                np.save(save_path, proba)


if __name__ == "__main__":
    # Receive dataset via argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str)
    parser.add_argument("k", nargs="?", type=list, help="the fold id (0 to K-1)")
    parser.add_argument(
        "tgt_rank",
        nargs="?",
        type=list,
        help="the rank of the target misclassification type",
    )
    parser.add_argument(
        "n", nargs="?", type=int, help="the factor for the number of neurons to fix"
    )
    parser.add_argument(
        "--misclf_type",
        type=str,
        help="the type of misclassification (src_tgt or tgt)",
        default="tgt",
    )
    parser.add_argument(
        "--fpfn",
        type=str,
        help="the type of misclassification (fp or fn)",
        default=None,
        choices=["fp", "fn"],
    )
    parser.add_argument(
        "--fl_method", type=str, help="the method used for FL", default="vdiff"
    )
    parser.add_argument("--run_all", action="store_true", help="run all settings")
    parser.add_argument(
        "--run_asc_desc", action="store_true", help="run only for asc and desc settings"
    )
    args = parser.parse_args()
    ds = args.ds
    k_list = args.k
    tgt_rank_list = args.tgt_rank
    n_list = args.n
    misclf_type = args.misclf_type
    fpfn = args.fpfn
    fl_method = args.fl_method
    run_all = args.run_all
    run_asc_desc = args.run_asc_desc

    # Error if run_all and run_asc_desc are specified at the same time
    assert not (
        run_all and run_asc_desc
    ), "run_all and run_asc_desc cannot be specified at the same time"

    if run_all:
        # Error if run_all is true but k and tgt_rank are specified
        assert (
            k_list is None and tgt_rank_list is None and n_list is None
        ), "run_all and k_list or tgt_rank_list or n_list cannot be specified at the same time"
        k_list = range(5)
        tgt_rank_list = range(1, 4)
        n_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 77, 109]
        fl_method_list = ["vdiff", "random"]
        misclf_type_list = ["all", "src_tgt", "tgt"]
        fpfn_list = [None, "fp", "fn"]
        for k, tgt_rank, n, fl_method, misclf_type, fpfn in product(
            k_list, tgt_rank_list, n_list, fl_method_list, misclf_type_list, fpfn_list
        ):
            if (misclf_type == "src_tgt" or misclf_type == "all") and fpfn is not None:
                continue
            main(ds, k, tgt_rank, n, fl_method, misclf_type, fpfn, run_all=run_all)
    elif run_asc_desc:
        assert (
            k_list is None and tgt_rank_list is None and n_list is None
        ), "run_all and k_list or tgt_rank_list or n_list cannot be specified at the same time"
        k_list = range(5)
        tgt_rank_list = range(1, 4)
        org_n_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 77, 109]
        n_list = [round(n / math.sqrt(2)) for n in org_n_list]
        # fl_method = ["vdiff_asc", "vdiff_desc"]
        fl_method = ["vdiff_desc"]
        misclf_type_list = ["all", "src_tgt", "tgt"]
        fpfn_list = [None, "fp", "fn"]
        for k, tgt_rank, n, misclf_type, fpfn in product(
            k_list, tgt_rank_list, n_list, misclf_type_list, fpfn_list
        ):
            if (misclf_type == "src_tgt" or misclf_type == "all") and fpfn is not None:
                continue
            main(ds, k, tgt_rank, n, fl_method, misclf_type, fpfn, run_all=run_all)
    else:
        assert (
            k_list is not None and tgt_rank_list is not None and n_list is not None
        ), "k_list and tgt_rank_list and n_list should be specified"
        for k, tgt_rank, n in zip(k_list, tgt_rank_list, n_list):
            main(ds, k, tgt_rank, n, fl_method, misclf_type, fpfn)
