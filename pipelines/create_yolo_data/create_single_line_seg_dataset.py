# %%
import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))
from argparse import ArgumentParser

from tqdm import tqdm
import yaml
from src.file_tools import write_text_file
from src.data_processing.visual_tasks import (
    bbox_xyxy_to_yolo_format, 
    coords_to_yolo_format,
)

from src.data_processing.florence import FlorenceSingleLineSegDataset


parser = ArgumentParser()
parser.add_argument("--source-data-dir", required=True)
parser.add_argument("--split-type", default="sbs")
args = parser.parse_args()

# args = parser.parse_args([
#     "--source-data-dir", "/Users/hoanghapham/Projects/vlm/data/pages/mixed",
#     "--split-type", "mixed"
# ])

SPLIT_TYPE = args.split_type
supported_split_types = ["sbs", "mixed"]
assert SPLIT_TYPE in supported_split_types, f"{SPLIT_TYPE} must be in: {supported_split_types}"

SOURCE_DATA_DIR = Path(args.source_data_dir)
YOLO_DATA_DIR   = PROJECT_DIR / f"data/yolo/" / SPLIT_TYPE / "single_line_seg"


# %%
# Prepare dirs

print(f"Create dirs in {YOLO_DATA_DIR}")
source_dirs = dict(
    train = SOURCE_DATA_DIR / "train",
    val = SOURCE_DATA_DIR / "val",
    test = SOURCE_DATA_DIR / "test"
)

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
    "names": {0: "textline"},
    "nc": 1
}

yaml.safe_dump(yolo_data_config, open(YOLO_DATA_DIR / "config.yaml", "w"))

# %%
# %%
# Write data

counts = dict(
    train = 0,
    val = 0,
    test = 0
)

for split, source_dir in source_dirs.items():
    print(f"Load data from {source_dir}")
    dataset = FlorenceSingleLineSegDataset(source_dir)
    dest_images_dir = YOLO_DATA_DIR / split / "images"
    dest_labels_dir = YOLO_DATA_DIR / split / "labels"

    # Iterate through datapoints
    print(f"Write data to {YOLO_DATA_DIR / split}")
    for data in tqdm(dataset):
        unique_key  = data["unique_key"]
        dest_image_path = dest_images_dir / (unique_key + ".png")

        if dest_image_path.exists():
            continue

        image       = data["image"]
        bbox        = data["bbox"]
        polygon     = data["polygon"]
        yolo_bbox   = bbox_xyxy_to_yolo_format(bbox, image.width, image.height, class_id=0)
        yolo_seg    = coords_to_yolo_format(polygon, image.width, image.height, class_id=0)

        # Write data
        dest_image_path = dest_images_dir / (unique_key + ".png")
        image.save(dest_image_path)
        
        dest_label_path = dest_labels_dir / (unique_key + ".txt")
        write_text_file(yolo_seg, dest_label_path)

        counts[split] += 1

print(f"Wrote {counts}")
    
# %%
