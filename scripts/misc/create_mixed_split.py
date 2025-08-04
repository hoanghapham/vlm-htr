# Script to create the mixed split
#%%
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np

from vlm.utils.file_tools import list_files, write_json_file
from vlm.data_processing.visual_tasks import IMAGE_EXTENSIONS

# Create curated sets
PROJECT_DIR = Path(__file__).parent.parent.parent
SOURCE_DATA_PATH = PROJECT_DIR / "data/raw/riksarkivet"

all_images = list_files(SOURCE_DATA_PATH, extensions=IMAGE_EXTENSIONS)

print(f"Total images: {len(all_images)}")

np.random.seed(42)

train_val_indices, test_indices = train_test_split(range(len(all_images)), test_size=0.1)

train_indices, val_indices = train_test_split(train_val_indices, test_size=0.1)

print("Train & val & test:\t", len(train_indices), len(val_indices), len(test_indices))

train_names = [all_images[idx].stem for idx in train_indices]
val_names   = [all_images[idx].stem for idx in val_indices]
test_names  = [all_images[idx].stem for idx in test_indices]


assert len(train_indices) + len(val_indices) + len(test_indices) == len(all_images)
assert len(train_names) + len(val_names) + len(test_names) == len(all_images)
assert len(set(train_names).intersection(set(val_names))) == 0
assert len(set(train_names).intersection(set(test_names))) == 0
assert len(set(val_names).intersection(set(test_names))) == 0

split_info = {
    "train": train_names,
    "val": val_names,
    "test": test_names
}

write_json_file(split_info, PROJECT_DIR / "configs/split_info/mixed.json")
