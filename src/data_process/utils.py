import sys
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from datasets import concatenate_datasets, load_from_disk

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.data_process.florence import RunningTextDataset


def load_(path_list: list[str | Path], data_class: Dataset | RunningTextDataset):
    dsets = []
    for path in path_list:
        dsets.append(load_from_disk(path))
    data = concatenate_datasets(dsets)
    return data_class(data)


def gen_split_indices(
    total_samples: int, 
    seed: int = 42, 
    train_ratio: float = 0.7, 
    val_ratio: float = 0.15, 
    test_ratio: float = 0.15
) -> tuple[list[int], list[int], list[int]]:
    np.random.seed(seed)
    all_indices = range(total_samples)

    train_indices = np.random.choice(all_indices, size=int(train_ratio * total_samples), replace=False)
    val_indices = np.random.choice(
        [idx for idx in all_indices if idx not in train_indices], 
        size = int(val_ratio * total_samples), 
        replace = False
    )
    test_indices = np.random.choice(
        [idx for idx in all_indices if idx not in np.concatenate([train_indices, val_indices])], 
        size = max(total_samples - len(train_indices) - len(val_indices), int(test_ratio * total_samples)),
        replace = False
    )

    return train_indices, val_indices, test_indices

