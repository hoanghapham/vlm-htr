#%%
import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from argparse import ArgumentParser
from src.data_process.utils import create_htr_split_info
from src.file_tools import read_json_file

#%%

parser = ArgumentParser()
parser.add_argument("--data-dir", required=True)
parser.add_argument("--regen", default="false")
args = parser.parse_args()

DATA_DIR = Path(args.data_dir)

split_info_path = DATA_DIR / "split_info.json"
if not split_info_path.exists() or args.regen == "true":
    create_htr_split_info(DATA_DIR, seed=42, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)


#%%

split_info = read_json_file(split_info_path)

print("Train, val, test size:", len(split_info["train"]), len(split_info["validation"]), len(split_info["test"]))
# %%
