import os
import numpy as np
from PIL import Image
import datasets
from datasets import load_dataset, load_from_disk, Dataset, DatasetInfo, DatasetDict
import argparse
from sklearn.model_selection import KFold
from tqdm import tqdm

def divide_train_repair(ori_train_dataset, num_fold=5):
    """
    divide the dataset into train and repair datasets.
    """
    train_fold_list, repair_fold_list = [], []
    kf = KFold(n_splits=num_fold, shuffle=True, random_state=777)
    for train_idx, repair_idx in kf.split(ori_train_dataset):
        train_fold = ori_train_dataset.select(train_idx)
        repair_fold = ori_train_dataset.select(repair_idx)
        train_fold_list.append(train_fold)
        repair_fold_list.append(repair_fold)
    return train_fold_list, repair_fold_list

def arr2img(arr):
    """
    (n, h, w, c)の形状のnumpy配列をn個のPIL.Imageのリストに変換して返す
    """
    imgs = []
    for a in tqdm(arr):
        imgs.append(Image.fromarray(a))
    return imgs

def get_sublist(original_list, severity):
    """severityに応じて適切な部分リストを取得する"""
    if severity == -1:
        return original_list # すべての要素を取得する
    elif severity >= 0 and severity <= 4:
        start_index = severity * 10000
        end_index = (severity + 1) * 10000 if severity < 4 else None
        return original_list[start_index:end_index]
    else:
        raise ValueError("severity must be an integer in the range 0 to 4 or -1")

parser = argparse.ArgumentParser(description='Dataset selector')
parser.add_argument('ds', type=str)
parser.add_argument('--severity', type=int, help="severity of corruption (integer from 0 to 4). when set to -1, treat all as one dataset.", default=-1)
parser.add_argument('--num_fold', type=int, help="the number for splitting the dataset", default=5)
args = parser.parse_args()
ds = args.ds
severity = args.severity
num_fold = args.num_fold
print(f"ds: {ds}")

# cifar10 # NOTE: We may not use this dataset.
if ds == "c10":
    ori_cifar10 = load_dataset("cifar10")
    ori_cifar10.save_to_disk("c10") # use load_from_disk when loading for the consistency
# cifar100
elif ds == "c100":
    ori_cifar100 = load_dataset("cifar100")
    ori_cifar100.save_to_disk("ori_c100") # use load_from_disk when loading for the consistency
    # divide the dataset into train and repair datasets
    train_fold_list, repair_fold_list = divide_train_repair(ori_cifar100["train"], num_fold)
    for k, (train_fold, repair_fold) in enumerate(zip(train_fold_list, repair_fold_list)):
        ds_dict = DatasetDict({"train": train_fold, "repair": repair_fold, "test": ori_cifar100["test"]})
        print(ds_dict)
        ds_dict.save_to_disk(f"c100_fold{k}")
# cifar10-c or cifar-100-c
elif ds == "c10c" or ds == "c100c":
    print(f"severity: {severity}")
    # original ds
    ori_ds = load_from_disk(ds.rstrip("c"))
    label_col = "label" if ds == "c10c" else "fine_label"
    # set the same info as the original
    info = DatasetInfo(
        features=datasets.Features(
            {
                "img": ori_ds["train"].features["img"],
                label_col: ori_ds["train"].features[label_col],
            }
        ),
    )
    # ds raw data path
    ds_raw_dir = os.path.join(f"{ds}/raw_data")
    # get .npy files
    npy_files = [f for f in os.listdir(ds_raw_dir) if f.endswith(".npy")]
    # make the dict (key=corruption name, val=corresponding image)
    labels = np.load(os.path.join(ds_raw_dir, "labels.npy"))
    labels = get_sublist(labels, severity) # slices only the portion corresponding to severity
    ds_dict = DatasetDict({})
    for npy_file in npy_files:
        key = npy_file.split(".npy")[0]
        if key == "labels":
            continue
        print(f"key: {key}")
        ds_arr = np.load(os.path.join(ds_raw_dir, npy_file))
        ds_arr = get_sublist(ds_arr, severity) # slices only the portion corresponding to severity
        print(f"ds_arr: {ds_arr.shape}")
        ds_dict[key] = Dataset.from_dict({"img": arr2img(ds_arr), label_col: labels}, info=info)
    print(ds_dict)
    ds_dict.save_to_disk(f"{ds}_severity{severity}") if severity != -1 else ds_dict.save_to_disk(ds)
else:
    raise NotImplementedError