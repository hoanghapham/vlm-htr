import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import unicodedata
import numpy as np
from torch.utils.data import Dataset
from datasets import concatenate_datasets, load_from_disk

from src.file_tools import read_json_file, write_json_file
from src.data_process.florence import RunningTextDataset


def create_dset_from_paths(path_list: list[str | Path], data_class: Dataset | RunningTextDataset):
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


def create_split_info(
    data_dir: Path | str,
    seed: int = 42,
    train_ratio: float = 0.7, 
    val_ratio: float = 0.15, 
    test_ratio: float = 0.15
):
    data_dir = Path(data_dir)
    split_info_path = Path(data_dir) / "split_info.json"
    page_paths = sorted([path for path in data_dir.glob("**/images/**/*")])

    train_indices, val_indices, test_indices = gen_split_indices(
        len(page_paths), 
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )
    split_info = {
        "train": [page_paths[idx].stem for idx in train_indices],
        "validation": [page_paths[idx].stem for idx in val_indices],
        "test": [page_paths[idx].stem for idx in test_indices]
    }
    write_json_file(split_info, split_info_path)


def normalize_name(s):
    return unicodedata.normalize('NFD', s)
