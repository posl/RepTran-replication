import os, sys, time
import argparse
import numpy as np
from itertools import product
import pickle
from utils.vit_util import get_misclf_info
from utils.constant import ViTExperiment

def main(ds_name, k, tgt_split):
    print(f"ds_name: {ds_name}, fold_id: {k}, tgt_split: {tgt_split}")

    # datasetの読み込み
    exp_obj = getattr(ViTExperiment, ds_name.replace("-", "_"))
    pretrained_dir = exp_obj.OUTPUT_DIR.format(k=k)
        
    if ds_name == "c100":
        num_classes = 100
    elif ds_name == "tiny-imagenet":
        num_classes = 200
    else:
        raise NotImplementedError(f"ds_name: {ds_name}")
    pred_out_dir = os.path.join(pretrained_dir, "pred_results", "PredictionOutput")
    pred_out_path = os.path.join(pred_out_dir, f"{tgt_split}_pred.pkl")

    with open(pred_out_path, "rb") as f:
        pred_out = pickle.load(f)
    pred_labels = pred_out.predictions.argmax(-1)
    true_labels = pred_out.label_ids
    mis_matrix, mis_ranking, mis_indices, met_dict = get_misclf_info(pred_labels, true_labels, num_classes)

    # mis_matrixはnpyで，それ以外はpklで保存
    save_dir = os.path.join(pretrained_dir, "misclf_info")
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f"{tgt_split}_mis_matrix.npy"), mis_matrix)
    with open(os.path.join(save_dir, f"{tgt_split}_mis_ranking.pkl"), "wb") as f:
        pickle.dump(mis_ranking, f)
    with open(os.path.join(save_dir, f"{tgt_split}_mis_indices.pkl"), "wb") as f:
        pickle.dump(mis_indices, f)
    with open(os.path.join(save_dir, f"{tgt_split}_met_dict.pkl"), "wb") as f:
        pickle.dump(met_dict, f)
    print("Summary of the misclassification info:")
    print(f"mis_matrix: {mis_matrix.shape}")
    print(f"total_mis: {mis_matrix.sum()}")

if __name__ == "__main__":
    # データセットをargparseで受け取る
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str)
    parser.add_argument('k', type=int, nargs="?", help="the fold id (0 to K-1)")
    parser.add_argument('--tgt_split', type=str, help="the target split name", default="repair")
    args = parser.parse_args()
    ds = args.ds
    k = args.k
    tgt_split = args.tgt_split
    if args.k is not None: # kが指定されている
        main(ds, k, tgt_split)
    else: # kが未指定
        for k, tgt_split in product(range(5), ["repair", "test"]):
            main(ds, k, tgt_split)