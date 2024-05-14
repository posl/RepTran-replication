import os
import torch


def get_device():
    """
    デバイス (cuda, or cpu) の取得
    
    Parameters
    ------------------
    
    Returns
    ------------------
    
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    return device

def get_corruption_types():
    # ds raw data path
    ds_raw_dir = "/src/dataset/c10c/raw_data"
    # get .npy files
    npy_files = [f for f in os.listdir(ds_raw_dir) if f.endswith(".npy")]
    # npy_filesのファイル名だけを保存
    corruption_types = [f.split(".npy")[0] for f in npy_files if f != "labels.npy"]
    return corruption_types