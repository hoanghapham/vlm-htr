import os
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from shutil import copy
from argparse import ArgumentParser

PROJECT_DIR = Path.cwd().parent
sys.path.append(str(PROJECT_DIR))

from src.file_tools import read_json_file, write_json_file, write_list_to_text_file
from src.data_process.xml import XMLParser


parser = ArgumentParser()
parser.add_argument("--input-dir", "-i", required=True)
parser.add_argument("--output-dir", "-o", required=True)
args = parser.parse_args()

INPUT_DIR = Path(args.input_dir)
OUTPUT_DIR = Path(args.output_dir)


split_info = read_json_file(INPUT_DIR / "split_info.json")

#%%
# Copy all train, val, test images to one folder
print("Copy files")

for split, img_xml_paths in split_info.items():
    for src_img, _ in tqdm(img_xml_paths, desc=split):
        
        dest_img = PROJECT_DIR / "data/yolo/all_images" / split / Path(src_img).name
        if not dest_img.parent.exists():
            dest_img.parent.mkdir(parents=True)

        copy(src_img, dest_img)
        


# %%

# Create symlink from all_images to 
print("Create symlinks")

for split, img_xml_paths in split_info.items():
    dest_dir = PROJECT_DIR / "data/yolo/regions/images/" / split
    if not dest_dir.exists():
        dest_dir.mkdir(parents=True)

    for img, _ in tqdm(img_xml_paths):
        os.symlink(
            img, 
            dest_dir / Path(img).name,
        )


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
        dest_dir = PROJECT_DIR / "data/yolo/regions/labels" / split
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
        dest_dir = PROJECT_DIR / "data/yolo/lines/labels" / split
        if not dest_dir.exists():
            dest_dir.mkdir(parents=True)

        write_list_to_text_file(yolo_bboxes, dest_dir / Path(img).with_suffix(".txt").name)

