import numpy as np
import torch
from pathlib import Path


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
    checkpoint_paths = sorted(path.glob("*.pt"))
    first_cp_info = torch.load(checkpoint_paths[0], weights_only=True, map_location=torch.device(device))

    best_loss = first_cp_info["loss"].item()
    best_state = first_cp_info

    for cp_path in checkpoint_paths:
        cp_info = torch.load(cp_path, weights_only=True, map_location=torch.device(device))
        if cp_info["loss"].item() < best_loss:
            best_loss = cp_info["loss"].item()
            best_state = cp_info

    return best_state
    

def load_last_checkpoint(path: Path, device: str) -> dict:

    last_cp_paths = list(reversed(sorted(path.glob("*.pt"))))
    if last_cp_paths:
        last_cp_path = last_cp_paths[0]

        last_cp_info = torch.load(last_cp_path, weights_only=True, map_location=torch.device(device))
        return last_cp_info
    else:
        return None