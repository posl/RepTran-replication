import os
import numpy as np
from PIL import Image
import datasets
from datasets import load_dataset, load_from_disk, Dataset, DatasetInfo, DatasetDict, ClassLabel, Features, Image as HFImage
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

def load_tiny_imagenet(tiny_imagenet_dir):
    """
    Tiny ImageNet のデータを Hugging Face Dataset の形式に変換
    """
    train_dir = os.path.join(tiny_imagenet_dir, "train")
    val_dir = os.path.join(tiny_imagenet_dir, "val")
    test_dir = os.path.join(tiny_imagenet_dir, "test")

    # クラス ID のリスト
    with open(os.path.join(tiny_imagenet_dir, "wnids.txt"), "r") as f:
        class_ids = [line.strip() for line in f.readlines()]
    
    class_to_label = {class_id: i for i, class_id in enumerate(class_ids)}
    
    # `words.txt` からクラス ID → ラベル名（英単語）の対応を取得
    class_id_to_name = {}
    with open(os.path.join(tiny_imagenet_dir, "words.txt"), "r") as f:
        for line in f.readlines():
            parts = line.strip().split("\t")
            if len(parts) == 2:
                class_id, label_name = parts
                if class_id in class_ids:
                    class_id_to_name[class_id] = label_name
    
    # クラス ID の順番を固定し、ラベル名リストを作成
    class_labels = [class_id_to_name[class_id] for class_id in class_ids]  # 実際のラベル名

    # ClassLabel オブジェクトを作成（整数 ID を持つが、`int2str()` でラベル名を取得可能）
    class_label_feature = ClassLabel(names=class_labels)
    
    # Features を適切に設定
    features = Features({
        "img": HFImage(),
        "label": class_label_feature  # ラベルは ID（整数）として保持
    })
    
    def load_images_and_labels(root_dir, class_to_label=None, has_labels=True):
        images, labels = [], []
        if has_labels:
            for class_id in os.listdir(root_dir):
                class_path = os.path.join(root_dir, class_id, "images")
                if not os.path.isdir(class_path):
                    continue
                for img_file in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_file)
                    images.append(Image.open(img_path).convert("RGB"))
                    labels.append(class_to_label[class_id])
            return images, labels
        else:
            # テストデータ（ラベルなし）
            images = [Image.open(os.path.join(root_dir, "images", img_file)).convert("RGB")
                      for img_file in os.listdir(os.path.join(root_dir, "images"))]
            return images, None

    # 訓練データの読み込み
    train_images, train_labels = load_images_and_labels(train_dir, class_to_label)

    # 検証データの読み込み
    val_annotations = os.path.join(val_dir, "val_annotations.txt")
    val_images, val_labels = [], []
    with open(val_annotations, "r") as f:
        for line in f.readlines():
            items = line.strip().split("\t")
            img_file, class_id = items[0], items[1]
            img_path = os.path.join(val_dir, "images", img_file)
            val_images.append(Image.open(img_path).convert("RGB"))
            val_labels.append(class_to_label[class_id])

    # テストデータの読み込み（ラベルなし）
    # test_images, _ = load_images_and_labels(test_dir, has_labels=False)

    return {
        "train": Dataset.from_dict({"img": train_images, "label": train_labels}, features=features),
        "repair": Dataset.from_dict({"img": val_images, "label": val_labels}, features=features),
        # "test": Dataset.from_dict({"img": test_images}, features=features),
    }

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
# Tiny ImageNet
elif ds == "tiny-imagenet":
    tiny_imagenet_dir = "/src/dataset/ori_tiny-imagenet-200"
    print(f"Processing Tiny ImageNet from {tiny_imagenet_dir} ...")
    ds_dict = DatasetDict(load_tiny_imagenet(tiny_imagenet_dir))
    print(ds_dict)
    ds_dict.save_to_disk("tiny-imagenet-200")
else:
    raise NotImplementedError