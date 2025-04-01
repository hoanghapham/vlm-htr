#%%
import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from src.logger import CustomLogger
from src.data_processing.utils import load_arrow_datasets
from src.file_tools import write_json_file

from tqdm import tqdm
#%%
# args = parser.parse_args([
#     "--data-dir", str(PROJECT_DIR / "data/cropped/mixed"),
#     "--model-name", "demo",
#     "--num-train-epochs", "5",
#     "--max-train-steps", "11",
#     "--batch-size", "2",
#     "--logging-interval", "3"
# ])


# Setup constant values
DATA_DIR            = PROJECT_DIR / "data/line_seg/mixed"

# Setup loggers
logger = CustomLogger("check_lines", log_to_local=True)

# Load model
logger.info("Load data")

# custom_question can be None
train_dataset   = load_arrow_datasets(DATA_DIR / "train")
val_dataset     = load_arrow_datasets(DATA_DIR / "val")
test_dataset    = load_arrow_datasets(DATA_DIR / "val")

# Check train

train_meta = {}
val_meta = {}
test_meta = {}

for data in tqdm(train_dataset):
    key = data["unique_key"]
    train_meta[key] = len(data["transcription"])


for data in tqdm(val_dataset):
    key = data["unique_key"]
    val_meta[key] = len(data["transcription"])

for data in tqdm(test_dataset):
    key = data["unique_key"]
    test_meta[key] = len(data["transcription"])


write_json_file(train_meta, "temp/train_meta.json")
write_json_file(val_meta, "temp/val_meta.json")
write_json_file(test_meta, "temp/test_meta.json")