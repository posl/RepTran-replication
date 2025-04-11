import os
import torch
import json

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

def json2dict(json_path):
    "jsonファイルを読み込んで辞書に変換する"
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d

def get_bottom3_keys_from_json(json_path):
    """
    指定されたJSONファイルから、値が小さい順にキーを3つ返す。
    
    Parameters:
        json_path (str): JSONファイルのパス
    
    Returns:
        List[str]: 値が小さい順の上位3つのキー
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return sorted(data, key=data.get)[:3]
