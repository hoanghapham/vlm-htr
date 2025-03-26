#%%
import os
import sys
from pathlib import Path

from tqdm import tqdm

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from src.file_tools import read_json_file, normalize_name


#%%

SOURCE_DIR = PROJECT_DIR / "data/processed/riksarkivet_line"
DEST_DIR = PROJECT_DIR / "data/florence/mixed"

if not DEST_DIR.exists():
    DEST_DIR.mkdir(parents=True)

split_page_names = read_json_file(PROJECT_DIR / "data/variants/mixed/split_page_names.json")


#%%
# Create symplink for florence

src_page_dirs = list(SOURCE_DIR.glob("*/*/"))

print(f"Total pages: {len(src_page_dirs)}")

#%%

for split, page_names in split_page_names.items():
    norm_page_names = [normalize_name(name) for name in page_names]
    dest_split_dir = DEST_DIR / split

    if not dest_split_dir.exists():
        dest_split_dir.mkdir(parents=True)

    for src_page in tqdm(src_page_dirs, desc=split):
        if normalize_name(src_page.stem) in norm_page_names:
            os.symlink(
                src_page,
                dest_split_dir / src_page.stem
            )
            