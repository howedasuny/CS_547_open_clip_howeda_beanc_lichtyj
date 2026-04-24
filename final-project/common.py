from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent

def get_device():
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("device:", device)
    return device