# %%
import sys
from pathlib import Path
from shutil import copy
PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))
from argparse import ArgumentParser

import yaml
from src.file_tools import read_json_file, write_list_to_text_file, normalize_name, list_files
from src.data_processing.visual_tasks import (
    bboxes_xyxy_to_yolo_format, 
    TextRegionBboxDataset, 
    TextLineBboxDataset, 
    IMAGE_EXTENSIONS
)
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("--split-type", default="sbs")
parser.add_argument("--object-class", default="region")
args = parser.parse_args()

SPLIT_TYPE      = args.split_type
OBJECT_CLASS    = args.object_class

supported_split_types = ["sbs", "mixed"]
assert SPLIT_TYPE in supported_split_types, f"{SPLIT_TYPE} must be in: {supported_split_types}"

supported_object_classes = ["region", "line"]
assert OBJECT_CLASS in supported_object_classes, f"{OBJECT_CLASS} must be in: {supported_object_classes}"


DATASET_NAME = f"{OBJECT_CLASS}_od"
SOURCE_DATA_DIR = PROJECT_DIR / "data/raw/riksarkivet"
YOLO_DATA_DIR   = PROJECT_DIR / f"data/yolo/" / SPLIT_TYPE / DATASET_NAME


# %%
# Prepare dirs

print(f"Create dirs in {YOLO_DATA_DIR}")

target_dirs = dict(
    train   = (YOLO_DATA_DIR / "train/images", YOLO_DATA_DIR / "train/labels"),
    val     = (YOLO_DATA_DIR / "val/images", YOLO_DATA_DIR / "val/labels"),
    test    = (YOLO_DATA_DIR / "test/images", YOLO_DATA_DIR / "test/labels"),
)

for key, paths in target_dirs.items():
    for path in paths:
        if not path.exists():
            path.mkdir(parents=True)

# %%
# Write config file

yolo_data_config = {
    "path": str(YOLO_DATA_DIR),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "names": {0: OBJECT_CLASS},
    "nc": 1
}

yaml.safe_dump(yolo_data_config, open(YOLO_DATA_DIR / "config.yaml", "w"))

# %%
# Prepare split info
split_info  = read_json_file(PROJECT_DIR / f"data/split_info/{SPLIT_TYPE}.json")

norm_train_names = [normalize_name(name) for name in split_info["train"]]
norm_val_names = [normalize_name(name) for name in split_info["val"]]
norm_test_names = [normalize_name(name) for name in split_info["test"]]

split_page_names = {
    "train": norm_train_names,
    "val": norm_val_names,
    "test": norm_test_names
}

#%%
# Load data

img_paths = list_files(SOURCE_DATA_DIR, IMAGE_EXTENSIONS) 
xml_paths = list_files(SOURCE_DATA_DIR, [".xml"])

if OBJECT_CLASS == "region":
    dataset = TextRegionBboxDataset(img_paths, xml_paths)
elif OBJECT_CLASS == "line":
    dataset = TextLineBboxDataset(img_paths, xml_paths)

# %%
# Write data

count_train = 0
count_val = 0
count_test = 0


# Iterate through datapoints
for data in tqdm(dataset):
    img_filename = normalize_name(data["img_filename"])
    image = data["image"]
    bboxes = data["bboxes"]
    yolo_bboxes = bboxes_xyxy_to_yolo_format(bboxes, image.width, image.height)
    source_img_path = data["img_path"]

    if img_filename in split_page_names["train"]:
        dest_images_dir, dest_labels_dir = target_dirs["train"]
        count_train += 1

    elif img_filename in split_page_names["val"]:
        dest_images_dir, dest_labels_dir = target_dirs["val"]
        count_val += 1

    elif img_filename in split_page_names["test"]:
        dest_images_dir, dest_labels_dir = target_dirs["test"]
        count_test += 1

    dest_image_path = dest_images_dir / Path(source_img_path).name
    dest_label_path = dest_labels_dir / (Path(source_img_path).stem + ".txt")
    
    copy(source_img_path, dest_image_path)
    write_list_to_text_file(yolo_bboxes, dest_label_path)

print(f"Wrote {count_train} train, {count_val} val, {count_test} test images.")
    
# %%
