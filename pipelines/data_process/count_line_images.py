#%%
import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from argparse import ArgumentParser

from src.data_processing.florence import FlorenceOCRDataset
from src.logger import CustomLogger

parser = ArgumentParser()
parser.add_argument("--data-dir", required=True)
args = parser.parse_args()

DATA_DIR = Path(args.data_dir)

train_dataset   = FlorenceOCRDataset(DATA_DIR / "train")
val_dataset     = FlorenceOCRDataset(DATA_DIR / "val")
test_dataset    = FlorenceOCRDataset(DATA_DIR / "test")

logger = CustomLogger(f"count_line_images__{DATA_DIR}", log_to_local=True)
logger.info(f"Number of train images: {len(train_dataset)}")
logger.info(f"Number of val images: {len(val_dataset)}")
logger.info(f"Number of test images: {len(test_dataset)}")