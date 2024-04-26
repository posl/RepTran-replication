import os
import numpy as np
from PIL import Image
import datasets
from datasets import load_dataset, load_from_disk, Dataset, DatasetInfo, DatasetDict
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Dataset selector')
parser.add_argument('ds', type=str)
args = parser.parse_args()
ds = args.ds
print(f"ds: {ds}")

def arr2img(arr):
    """
    (n, h, w, c)の形状のnumpy配列をn個のPIL.Imageのリストに変換して返す
    """
    imgs = []
    for a in tqdm(arr):
        imgs.append(Image.fromarray(a))
    return imgs

# cifar10
if ds == "c10":
    cifar10 = load_dataset("cifar10")
    cifar10.save_to_disk("c10") # use load_from_disk when loading for the consistency
# cifar100
elif ds == "c100":
    cifar100 = load_dataset("cifar100")
    cifar100.save_to_disk("c100") # use load_from_disk when loading for the consistency
# cifar10-c
elif ds == "c10c":
    # original c10
    cifar10 = load_from_disk("c10")
    # set the same info as the original
    info = DatasetInfo(
        features=datasets.Features(
            {
                "img": cifar10["train"].features["img"],
                "label": cifar10["train"].features["label"],
            }
        ),
    )
    # c10c raw data path
    c10c_raw_dir = os.path.join("c10c/raw_data")
    # get .npy files
    npy_files = [f for f in os.listdir(c10c_raw_dir) if f.endswith(".npy")]
    # make the dict (key=corruption name, val=corresponding image)
    labels = np.load(os.path.join(c10c_raw_dir, "labels.npy"))
    ds_dict = DatasetDict({})
    for npy_file in npy_files:
        key = npy_file.split(".npy")[0]
        if key == "labels":
            continue
        print(f"key: {key}")
        ds_arr = np.load(os.path.join(c10c_raw_dir, npy_file))
        ds_dict[key] = Dataset.from_dict({"img": arr2img(ds_arr), "label": labels}, info=info)
    print(ds_dict)
    ds_dict.save_to_disk("c10c")
# cifar100-c
elif ds == "c100c":
    # original c100
    cifar100 = load_from_disk("c100")
    # set the same info as the original
    info = DatasetInfo(
        features=datasets.Features(
            {
                "img": cifar100["train"].features["img"],
                "label": cifar100["train"].features["fine_label"],
            }
        ),
    )
    # c100c raw data path
    c100c_raw_dir = os.path.join("c100c/raw_data")
    # get .npy files
    npy_files = [f for f in os.listdir(c100c_raw_dir) if f.endswith(".npy")]
    # make the dict (key=corruption name, val=corresponding image)
    labels = np.load(os.path.join(c100c_raw_dir, "labels.npy"))
    ds_dict = DatasetDict({})
    for npy_file in npy_files:
        key = npy_file.split(".npy")[0]
        if key == "labels":
            continue
        print(f"key: {key}")
        ds_arr = np.load(os.path.join(c100c_raw_dir, npy_file))
        ds_dict[key] = Dataset.from_dict({"img": arr2img(ds_arr), "label": labels}, info=info)
    print(ds_dict)
    ds_dict.save_to_disk("c100c")
else:
    raise NotImplementedError