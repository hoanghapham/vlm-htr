# Script to create symlinks for datasets created from the HTRDatasetBuilder
#%%
import os
import sys
from pathlib import Path

from tqdm import tqdm
from argparse import ArgumentParser

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from src.file_tools import read_json_file, normalize_name, list_files
from src.data_processing.visual_tasks import IMAGE_EXTENSIONS

#%%

parser = ArgumentParser()
parser.add_argument("--source-data-dir", required=True)
parser.add_argument("--dest-data-dir", required=True)
parser.add_argument("--split-info-path", required=True)
args = parser.parse_args()

SOURCE_DATA_DIR = Path(args.source_data_dir)
DEST_DATA_DIR   = Path(args.dest_data_dir)
SPLIT_INFO_PATH = Path(args.split_info_path)

if not DEST_DATA_DIR.exists():
    DEST_DATA_DIR.mkdir(parents=True)

split_info = read_json_file(SPLIT_INFO_PATH)


#%%
# Create symplink for florence

img_paths = list_files(SOURCE_DATA_DIR, IMAGE_EXTENSIONS)
xml_paths = list_files(SOURCE_DATA_DIR, [".xml"])


img_names = [path.stem for path in img_paths]
xml_names = [path.stem for path in xml_paths]
matched = set(img_names).intersection(set(xml_names))

assert len(img_paths) == len(xml_paths) == matched > 0, f"Length mismatch: {len(img_paths)} images, {len(xml_paths)} XML files, {len(matched)} matches"

all_imgs_xmls = list(zip(img_paths, xml_paths))

print(f"Total pages: {len(all_imgs_xmls)}")

#%%

counts = {
    "train": 0,
    "val": 0,
    "test": 0
}

for split, page_names in split_info.items():
    norm_page_names = [normalize_name(name) for name in page_names]
    dest_split_dir = DEST_DATA_DIR / split

    if not dest_split_dir.exists():
        dest_split_dir.mkdir(parents=True)
    
    for src_img, src_xml in tqdm(all_imgs_xmls, desc=split):

        if normalize_name(src_img.stem) in norm_page_names:
            src_img_abs = src_img.resolve()
            dst_img_abs = (dest_split_dir / src_img.name).resolve()
            os.symlink(src_img_abs, dst_img_abs)

            src_xml_abs = src_xml.resolve()
            dst_xml_abs = (dest_split_dir / src_xml.name).resolve()
            os.symlink(src_xml_abs, dst_xml_abs)
            
            counts[split] += 1

print("Created symlinks:", counts)
missed = len(all_imgs_xmls) - sum(counts.values())
print("Missed:", missed)
