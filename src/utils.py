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
