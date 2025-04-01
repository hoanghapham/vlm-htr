#%%
import os
import sys
from pathlib import Path

from tqdm import tqdm
from argparse import ArgumentParser

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from src.file_tools import read_json_file, normalize_name


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

src_page_dirs = [path for path in sorted(SOURCE_DATA_DIR.glob("*")) if path.is_dir()]

print(f"Total pages: {len(src_page_dirs)}")

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

    for src_page in tqdm(src_page_dirs, desc=split):
        if normalize_name(src_page.stem) in norm_page_names:
            os.symlink(
                src_page,
                dest_split_dir / src_page.stem
            )
            
            counts[split] += 1

print("Created symlinks:", counts)