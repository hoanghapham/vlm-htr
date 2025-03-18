#%%
import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

import torch
from torch.utils.data import Dataset, DataLoader
from htrflow.evaluate import CER, WER, BagOfWords

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from src.tasks.utils import create_dset_from_paths, create_split_info
from src.tasks.running_text import RunningTextDataset
from src.file_tools import write_json_file, read_json_file
from src.train import load_best_checkpoint
from src.logger import CustomLogger

# REMOTE_MODEL_PATH = "Riksarkivet/trocr-base-handwritten-hist-swe-2"
REMOTE_MODEL_PATH = "microsoft/trocr-base-handwritten"
LOCAL_MODEL_PATH = PROJECT_DIR / "model/trocr_base__ft_htr_line"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = PROJECT_DIR / "data/polis_line"

logger = CustomLogger("trocr_base__ft_htr_line", log_to_local=True)

#%%

split_info_path = DATA_DIR / "split_info.json"
page_path_list = sorted([path for path in DATA_DIR.glob("*") if path.is_dir()])

if not split_info_path.exists():
    create_split_info(DATA_DIR, seed=42)

split_info = read_json_file(split_info_path)

train_paths = [path for path in page_path_list if path.stem in split_info["train"]]
val_paths = [path for path in page_path_list if path.stem in split_info["validation"]]

# Create dataset object
train_dataset = create_dset_from_paths(train_paths, RunningTextDataset)
val_dataset = create_dset_from_paths(val_paths, RunningTextDataset)


#%%
# Load model
processor = TrOCRProcessor.from_pretrained(REMOTE_MODEL_PATH)
model = VisionEncoderDecoderModel.from_pretrained(REMOTE_MODEL_PATH).to(device)
model.eval()

#%%

idx = 19

pixel_values = processor(images=train_dataset[idx]["image"], return_tensors="pt").pixel_values

generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(train_dataset[idx]["answer"])
print(generated_text)
# %%


best_state = load_best_checkpoint(LOCAL_MODEL_PATH, DEVICE)
model.load_state_dict(best_state["model_state_dict"])
best_epoch = best_state["epoch"]
best_train_loss = best_state["avg_train_loss"]
best_val_loss = best_state["avg_val_loss"]

logger.info(f"Best checkpoint: epoch {best_epoch}, train loss: {best_train_loss:.4f}, validation loss: {best_val_loss}")
