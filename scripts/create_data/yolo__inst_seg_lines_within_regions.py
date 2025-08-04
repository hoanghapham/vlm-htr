# %%
from argparse import ArgumentParser
import yaml
from vlm.utils.file_tools import read_json_file, write_list_to_text_file, normalize_name
from vlm.data_processing.yolo import polygon_to_yolo_format
from datasets import load_from_disk
from tqdm import tqdm


parser = ArgumentParser()
parser.add_argument("--split-type", default="mixed")
parser.add_argument("--debug", required=False, default="false")
args = parser.parse_args()

# args = parser.parse_args([
#     "--split-type", "mixed",
#     "--debug", "true"
# ])

SPLIT_TYPE      = args.split_type
DEBUG           = args.debug == "true"

dataset_name    = "inst_seg_lines_within_regions"
SOURCE_DATA_DIR = PROJECT_DIR / "data/processed/riksarkivet" / dataset_name
YOLO_DATA_DIR   = PROJECT_DIR / f"data/yolo/{SPLIT_TYPE}/{dataset_name}"


# Load data
dsets = []
print("Load all datasets")
dir_paths   = sorted([path for path in SOURCE_DATA_DIR.iterdir() if path.is_dir()])
split_info  = read_json_file(PROJECT_DIR / f"configs/split_info/{SPLIT_TYPE}.json")


# %%
# Prepare dirs

print(f"Create dirs in {YOLO_DATA_DIR}")
train_dest = YOLO_DATA_DIR / "train"
val_dest = YOLO_DATA_DIR / "val"
test_dest = YOLO_DATA_DIR / "test"

train_image_dir = YOLO_DATA_DIR / "train/images"
val_image_dir = YOLO_DATA_DIR / "val/images"
test_image_dir = YOLO_DATA_DIR / "test/images"

train_label_dir = YOLO_DATA_DIR / "train/labels"
val_label_dir = YOLO_DATA_DIR / "val/labels"
test_label_dir = YOLO_DATA_DIR / "test/labels"

if not train_image_dir.exists():
    train_image_dir.mkdir(parents=True)

if not val_image_dir.exists():
    val_image_dir.mkdir(parents=True)

if not test_image_dir.exists():
    test_image_dir.mkdir(parents=True)

if not train_label_dir.exists():
    train_label_dir.mkdir(parents=True)

if not val_label_dir.exists():  
    val_label_dir.mkdir(parents=True)

if not test_label_dir.exists():
    test_label_dir.mkdir(parents=True)

# %%
# Write config file

yolo_data_config = {
    "path": str(YOLO_DATA_DIR),
    "train": "train/images",
    "val": "val/images",
    "test": "test/images",
    "names": {0: "line"},
    "nc": 1
}

yaml.safe_dump(yolo_data_config, open(YOLO_DATA_DIR / "config.yaml", "w"))

# %%
# Prepare normalized names

norm_train_names = [normalize_name(name) for name in split_info["train"]]
norm_val_names = [normalize_name(name) for name in split_info["val"]]
norm_test_names = [normalize_name(name) for name in split_info["test"]]

split_page_names = {
    "train": norm_train_names,
    "val": norm_val_names,
    "test": norm_test_names
}

# %%
# Write data
count_train = 0
count_val = 0
count_test = 0


for idx, path in enumerate(dir_paths):
    print(f"Process dataset {idx}/{len(dir_paths)}")
    try:
        dataset = load_from_disk(path)
    except Exception as e:
        print(e)
        continue

    # Iterate through datapoints
    for data in tqdm(dataset):
        img_filename = normalize_name(data["img_filename"])
        image = data["image"]
        annotations = data["annotations"]
        polygons = []
        yolo_annotations = []

        for ann in annotations:
            polygons.append(ann["polygon"])
            yolo_annotations.append(polygon_to_yolo_format(ann["polygon"], image.width, image.height))

        if img_filename in split_page_names["train"]:
            image.save(train_dest / "images/" / f"{img_filename}.png")
            write_list_to_text_file(yolo_annotations, train_dest / "labels/" / f"{img_filename}.txt")
            count_train += 1

        elif img_filename in split_page_names["val"]:
            image.save(val_dest / "images/" / f"{img_filename}.png")
            write_list_to_text_file(yolo_annotations, val_dest / "labels/" / f"{img_filename}.txt")
            count_val += 1

        elif img_filename in split_page_names["test"]:
            image.save(test_dest / "images/" / f"{img_filename}.png")
            write_list_to_text_file(yolo_annotations, test_dest / "labels/" / f"{img_filename}.txt")
            count_test += 1
    if DEBUG:
        break

print(f"Wrote {count_train} train, {count_val} val, {count_test} test images.")
    
# %%
