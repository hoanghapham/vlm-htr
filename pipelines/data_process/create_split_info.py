#%%
import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from argparse import ArgumentParser
from src.data_process.utils import create_split_info
from src.file_tools import read_json_file

#%%

parser = ArgumentParser()
parser.add_argument("--data-dir", required=True)
args = parser.parse_args()

DATA_DIR = Path(args.data_dir)

split_info_path = DATA_DIR / "split_info.json"
create_split_info(DATA_DIR, seed=42, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)


#%%

split_info = read_json_file("/Users/hoanghapham/Projects/thesis-data/riksarkivet/split_info.json")

print(len(split_info["train"]), len(split_info["validation"]), len(split_info["test"]))
# %%
