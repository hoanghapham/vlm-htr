#%%
import sys
from pathlib import Path
from tqdm import tqdm
from shutil import copy
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from src.file_tools import list_files

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


train_val_files = []

for dirname in train_val_dirs:
    images = list_files(riks_path / dirname, extensions=[".tif", ".jpg"])
    xmls = list_files(riks_path / dirname, extensions=[".xml"])
    train_val_files += list(zip(images, xmls))


test_files = []

for dirname in test_dirs:
    images = list_files(riks_path / dirname, extensions=[".tif", ".jpg"])
    xmls = list_files(riks_path / dirname, extensions=[".xml"])
    test_files += list(zip(images, xmls))

print("Train-val & test:\t", len(train_val_files), len(test_files))

train_indices, val_indices = train_test_split(range(len(train_val_files)), test_size=0.07)

print("Train & val:\t\t", len(train_indices), len(val_indices))

assert len(train_indices) + len(val_indices) == len(train_val_files)
# Copy train, val, test files to destination
curated_path = PROJECT_DIR / "data/variants/curated"

split_info = {
    "train": list(train_indices),
    "val": list(val_indices),
    "test": list(range(len(test_files)))
}

for split, indices in split_info.items():
    images_dir = curated_path / "images" / split
    page_xmls_dir = curated_path / "page_xmls" / split

    if not images_dir.exists():
        images_dir.mkdir(parents=True)

    if not page_xmls_dir.exists():
        page_xmls_dir.mkdir(parents=True)

    # Copy train & val
    if split in ["train", "val"]:
        for idx in tqdm(indices, desc=split):
            copy(train_val_files[idx][0], images_dir / train_val_files[idx][0].name)
            copy(train_val_files[idx][1], page_xmls_dir / train_val_files[idx][1].name)
    else:
        for idx in tqdm(indices, desc=split):
            copy(test_files[idx][0], images_dir / test_files[idx][0].name)
            copy(test_files[idx][1], page_xmls_dir / test_files[idx][1].name)

