# %%
import os
import shutil
import yaml
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm

from vlm.utils.file_tools import read_json_file, write_list_to_text_file, normalize_name
from vlm.data_processing.yolo import YOLOPageRegionODDataset, YOLOPageLineODDataset, YOLORegionLineODDataset

parser = ArgumentParser()
parser.add_argument("--split-type", default="sbs")
parser.add_argument("--task", required=True, choices=["page__region_od", "page__line_od", "region__line_od"])
parser.add_argument("--debug", required=False, default="false")
# args = parser.parse_args()

args = parser.parse_args([
    "--split-type", "mixed",
    "--task", "region__line_od",    
    "--debug", "true"
])

SPLIT_TYPE  = args.split_type
TASK        = args.task
DEBUG       = args.debug == "true"

supported_split_types = ["sbs", "mixed"]
assert SPLIT_TYPE in supported_split_types, f"{SPLIT_TYPE} must be in: {supported_split_types}"

supported_tasks = ["page__region_od", "page__line_od", "region__line_od"]
assert TASK in supported_tasks, f"{TASK} must be in: {supported_tasks}"

PROJECT_DIR = Path(__file__).parent.parent.parent
SOURCE_DATA_DIR = PROJECT_DIR / "data/raw/riksarkivet"
YOLO_DATA_DIR   = PROJECT_DIR / f"data/yolo/" / SPLIT_TYPE / TASK


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
    "train": "train/images",
    "val": "val/images",
    "test": "test/images",
    "names": {0: TASK},
    "nc": 1
}

yaml.safe_dump(yolo_data_config, open(YOLO_DATA_DIR / "config.yaml", "w"))

# %%
# Prepare split info
split_info  = read_json_file(PROJECT_DIR / f"configs/split_info/{SPLIT_TYPE}.json")

split_page_names = {
    "train": [normalize_name(name) for name in split_info["train"]],
    "val": [normalize_name(name) for name in split_info["val"]],
    "test": [normalize_name(name) for name in split_info["test"]]
}

#%%
# Load data

if TASK == "page__region_od":
    dataset = YOLOPageRegionODDataset(SOURCE_DATA_DIR)
elif TASK == "page__line_od":
    dataset = YOLOPageLineODDataset(SOURCE_DATA_DIR)
elif TASK == "region__line_od":
    dataset = YOLORegionLineODDataset(SOURCE_DATA_DIR)

#%%
# from src.visualization import draw_bboxes_xyxy
# draw_bboxes_xyxy(dataset[0]["image"], dataset[0]["bboxes"])

# %%
# Write data
count_train = 0
count_val = 0
count_test = 0

# Iterate through datapoints
for data in tqdm(dataset, total=len(dataset), desc=f"Write data to {YOLO_DATA_DIR}"):
    img_filename = normalize_name(data["img_filename"])
    img_volume = data["img_volume"]

    # Determine write path
    if img_filename in split_page_names["train"]:
        dest_images_dir, dest_labels_dir = target_dirs["train"]
        count_train += 1

    elif img_filename in split_page_names["val"]:
        dest_images_dir, dest_labels_dir = target_dirs["val"]
        count_val += 1

    elif img_filename in split_page_names["test"]:
        dest_images_dir, dest_labels_dir = target_dirs["test"]
        count_test += 1

    unique_key = data["unique_key"]
    dest_image_path = dest_images_dir / f"{unique_key}.png"
    dest_label_path = dest_labels_dir / f"{unique_key}.txt"

    full_img_path = dest_images_dir / f"{img_volume}_{img_filename}.png"
    full_label_path = dest_labels_dir / f"{img_volume}_{img_filename}.txt"

    # Temp fix, should be removed
    if full_label_path.exists() and full_label_path.exists():
        shutil.move(full_img_path, dest_image_path)
        shutil.move(full_label_path, dest_label_path)
        os.remove(full_img_path)
        os.remove(full_label_path)
        continue

    # Write data  
    image = data["image"]
    bboxes = data["bboxes"]
    yolo_bboxes = data["yolo_bboxes"]
    source_img_path = data["img_path"]

    image.save(dest_image_path)
    write_list_to_text_file(yolo_bboxes, dest_label_path)

    if DEBUG:
        break

print(f"Wrote {count_train} train, {count_val} val, {count_test} test images.")
    
# %%
