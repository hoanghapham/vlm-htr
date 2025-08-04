#%%
from pathlib import Path
from argparse import ArgumentParser

from vlm.data_processing.florence import FlorenceRegionLineODDataset
from vlm.utils.logger import CustomLogger

parser = ArgumentParser()
parser.add_argument("--data-dir", required=True)
args = parser.parse_args()

DATA_DIR = Path(args.data_dir)

train_dataset   = FlorenceRegionLineODDataset(DATA_DIR / "train")
val_dataset     = FlorenceRegionLineODDataset(DATA_DIR / "val")
test_dataset    = FlorenceRegionLineODDataset(DATA_DIR / "test")

logger = CustomLogger(f"count_region_images", log_to_local=True)
logger.info(f"Number of train images: {len(train_dataset)}")
logger.info(f"Number of val images: {len(val_dataset)}")
logger.info(f"Number of test images: {len(test_dataset)}")

logger.info(f"Valid train images: {len(train_dataset.xml_paths)}")
logger.info(f"Valid val images: {len(val_dataset.xml_paths)}")
logger.info(f"Valid test images: {len(test_dataset.xml_paths)}")