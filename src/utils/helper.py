import os
import torch
import json

def get_device():
    """
    Get device (cuda, or cpu)
    
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
    # Save only the filenames of npy_files
    corruption_types = [f.split(".npy")[0] for f in npy_files if f != "labels.npy"]
    return corruption_types

def json2dict(json_path):
    "Load JSON file and convert to dictionary"
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d

def get_bottom3_keys_from_json(json_path):
    """
    Return 3 keys from the specified JSON file in ascending order of values.
    
    Parameters:
        json_path (str): Path to JSON file
    
    Returns:
        List[str]: Top 3 keys in ascending order of values
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return sorted(data, key=data.get)[:3]
