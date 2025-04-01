#%%
import sys
from pathlib import Path
from tqdm import tqdm
from shutil import copy
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from src.file_tools import list_files, write_json_file
from src.data_processing.visual_tasks import IMAGE_EXTENSIONS

# Create curated sets

riks_path = PROJECT_DIR / "data/raw/riksarkivet"

train_val_dirs = [
    "trolldoms",
    "svea_hovratt",
    "bergskollegium_rel",
    "poliskammare",
]

test_dirs = [
    "jonkopings_radhusratts",
    "gota_hovratt",
    "bergskollegium_adv",
    "frihetstidens",
    "krigshovrattens",
]


train_val_names = []

for dirname in train_val_dirs:
    images  = list_files(riks_path / dirname, extensions=IMAGE_EXTENSIONS)
    names   = [path.stem for path in images]
    train_val_names += names


test_names = []

for dirname in test_dirs:
    images  = list_files(riks_path / dirname, extensions=IMAGE_EXTENSIONS)
    names   = [path.stem for path in images]
    test_names += names

print("Train-val & test:\t", len(train_val_names), len(test_names))

# Split
train_indices, val_indices = train_test_split(range(len(train_val_names)), test_size=0.07)

print("Train & val:\t\t", len(train_indices), len(val_indices))


assert len(train_indices) + len(val_indices) == len(train_val_names)

train_names = [train_val_names[idx] for idx in train_indices]
val_names   = [train_val_names[idx] for idx in val_indices]

split_info = {
    "train": train_names,
    "val": val_names,
    "test": test_names
}

write_json_file(split_info, PROJECT_DIR / "data/split_info/sbs.json")
