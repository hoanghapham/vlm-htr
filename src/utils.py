import os, sys
sys.path.append(os.path.dirname(__file__))

import numpy as np
import torch
from pathlib import Path
from file_tools import read_json_file


def gen_split_indices(total_samples: int, seed: int = 42):

    np.random.seed(seed)

    train_indices = np.random.choice(range(total_samples), size=int(0.7 * total_samples), replace=False)
    
    val_indices = np.random.choice(
        [idx for idx in range(total_samples) if idx not in train_indices], 
        size=int(0.15 * total_samples), replace=False
    )
    
    test_inidices = np.random.choice(
        [idx for idx in range(total_samples) if idx not in np.concatenate([train_indices, val_indices])], 
        size = total_samples - len(train_indices) - len(val_indices),
        replace=False
    )

    return train_indices, val_indices, test_inidices

def load_best_checkpoint(path: Path, device: str) -> dict:
    cp_metric_paths = sorted(path.glob("*.json"))
    cp_state_paths = sorted(path.glob("*.pt"))

    best_loss = float("inf")
    best_epoch = 0

    for cp_path in cp_metric_paths:
        cp_metric = read_json_file(cp_path)
        if cp_metric["loss"].item() < best_loss:
            best_loss = cp_metric["loss"]
            best_epoch = cp_metric["epoch"]
    
    best_state = torch.load(cp_state_paths[best_epoch], weights_only=True, map_location=torch.device(device))
    best_state['epoch'] = best_epoch
    best_state["loss"] = best_loss

    return best_state
    

def load_last_checkpoint(path: Path, device: str) -> dict:

    last_cp_state_path = list(reversed(sorted(path.glob("*.pt"))))
    last_cp_metric_path = list(reversed(sorted(path.glob("*.json"))))

    if last_cp_state_path:
        last_cp_path = last_cp_state_path[0]
        last_cp_metric = read_json_file(last_cp_metric_path[0])
        last_cp_state = torch.load(last_cp_path, weights_only=True, map_location=torch.device(device))
        last_cp_state.update(last_cp_metric)
        return last_cp_state
    else:
        return None