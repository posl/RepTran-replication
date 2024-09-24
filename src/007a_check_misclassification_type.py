import os, sys, time
import argparse
import numpy as np
import pickle
from utils.vit_util import get_misclf_info
from utils.constant import ViTExperiment

if __name__ == '__main__':
    # データセットをargparseで受け取る
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str)
    parser.add_argument('k', type=int, help="the fold id (0 to K-1)")
    args = parser.parse_args()
    ds_name = args.ds
    k = args.k
    print(f"ds_name: {ds_name}, fold_id: {k}")

    tgt_split = "repair"
    if ds_name == "c100":
        num_classes = 100
    else:
        NotImplemented, f"ds_name: {ds_name}"
    pretrained_dir = getattr(ViTExperiment, ds_name).OUTPUT_DIR.format(k=k)
    pred_out_dir = os.path.join(pretrained_dir, "pred_results", "PredictionOutput")
    pred_out_path = os.path.join(pred_out_dir, f"{tgt_split}_pred.pkl")

    with open(pred_out_path, "rb") as f:
        pred_out = pickle.load(f)
    pred_labels = pred_out.predictions.argmax(-1)
    true_labels = pred_out.label_ids
    mis_matrix, mis_ranking, mis_indices = get_misclf_info(pred_labels, true_labels, num_classes)
    # mis_matrixはnpyで，それ以外はpklで保存
    save_dir = os.path.join(pretrained_dir, "misclf_info")
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f"{tgt_split}_mis_matrix.npy"), mis_matrix)
    with open(os.path.join(save_dir, f"{tgt_split}_mis_ranking.pkl"), "wb") as f:
        pickle.dump(mis_ranking, f)
    with open(os.path.join(save_dir, f"{tgt_split}_mis_indices.pkl"), "wb") as f:
        pickle.dump(mis_indices, f)
    print("Summary of the misclassification info:")
    print(f"mis_matrix: {mis_matrix.shape}")
    print(f"total_mis: {mis_matrix.sum()}")
    for i, j, mis in mis_ranking[:10]:
        print(f"{i} -> {j}: {mis} / {mis_matrix.sum()} = {100 * mis / mis_matrix.sum():.2f} %")
        print(f"mis_indices: {mis_indices[i][j]}")