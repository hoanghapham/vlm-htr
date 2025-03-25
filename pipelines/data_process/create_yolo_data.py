import os
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from shutil import copy
from argparse import ArgumentParser

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from src.file_tools import read_json_file, write_json_file, write_list_to_text_file
from src.data_process.xml import XMLParser


parser = ArgumentParser()
parser.add_argument("--input-dir", "-i", required=True)
parser.add_argument("--output-dir", "-o", required=True)
parser.add_argument("--copy-files", "-cp", default="true")
args = parser.parse_args()

INPUT_DIR = Path(args.input_dir)
OUTPUT_DIR = Path(args.output_dir)
COPY_FILES = args.copy_files == "true"

split_info = read_json_file(INPUT_DIR / "split_info.json")

#%%
# Copy all train, val, test images to one folder

if COPY_FILES:
    print("Copy files")

split_img_paths = {}

for split, img_xml_paths in split_info.items():
    split_img_paths[split] = []

    for src_img, _ in tqdm(img_xml_paths, desc=split):
        
        dest_img = OUTPUT_DIR / split / Path(src_img).name
        if not dest_img.parent.exists():
            dest_img.parent.mkdir(parents=True)

        if COPY_FILES:
            copy(src_img, dest_img)

        split_img_paths[split].append(str(dest_img))
        
# write_json_file(split_img_paths, OUTPUT_DIR / "split_info.json")

# %%

# Create symlink from all_images to 
print("Create symlinks")
tasks = ["line_detection", "region_detection"]

for task in tasks:

    for split, img_paths in split_img_paths.items():
        dest_dir = PROJECT_DIR / f"data/yolo/{task}/images/{split}"
        if not dest_dir.exists():
            dest_dir.mkdir(parents=True)

        for img in tqdm(img_paths, desc=f"{task} - {split}"):
            os.symlink(img, dest_dir / Path(img).name)


# %%

parser = XMLParser()

def convert_to_yolo_format(bboxes: list[tuple], img_width, img_height, class_id=0):
    yolo_annotations = []
    
    for (xmin, ymin, xmax, ymax) in bboxes:
        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")
    
    return yolo_annotations


#%%
# Create regions data
print("Create region labels")

for split, img_xml_paths in split_info.items():
    for img, xml in tqdm(img_xml_paths, desc=split):
        # Get image info
        image = Image.open(img)

        # Get bbox info
        regions = parser.get_regions(xml)
        bboxes = [reg["bbox"] for reg in regions]
        yolo_bboxes = convert_to_yolo_format(bboxes, img_width=image.width, img_height=image.height, class_id=0)

        # Write
        dest_dir = PROJECT_DIR / "data/yolo/region_detection/labels" / split
        if not dest_dir.exists():
            dest_dir.mkdir(parents=True)

        write_list_to_text_file(yolo_bboxes, dest_dir / Path(img).with_suffix(".txt").name)

# %%
# Create lines data
print("Create line labels")

for split, img_xml_paths in split_info.items():
    for img, xml in tqdm(img_xml_paths, desc=split):
        # Get image info
        image = Image.open(img)

        # Get bbox info
        lines = parser.get_lines(xml)
        bboxes = [line["bbox"] for line in lines]
        yolo_bboxes = convert_to_yolo_format(bboxes, img_width=image.width, img_height=image.height, class_id=1)

        # Write
        dest_dir = PROJECT_DIR / "data/yolo/line_detection/labels" / split
        if not dest_dir.exists():
            dest_dir.mkdir(parents=True)

        write_list_to_text_file(yolo_bboxes, dest_dir / Path(img).with_suffix(".txt").name)

