#%%
import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).parent.parent  
sys.path.append(str(PROJECT_DIR))

import torch
from torch.utils.data import Dataset, DataLoader
from htrflow.evaluate import CER, WER, BagOfWords

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from src.htr_tools import load_from_disk, create_dset_from_paths
from src.file_tools import write_json_file, read_json_file
from src.utils import gen_split_indices

TROCR_PATH = "microsoft/trocr-base-handwritten"
REMOTE_MODEL_PATH = "Riksarkivet/trocr-base-handwritten-hist-swe-2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = PROJECT_DIR / "data/polis_line"

#%%

split_info_path = DATA_DIR / "split_info.json"
page_path_list = sorted([path for path in DATA_DIR.glob("*") if path.is_dir()])

if not split_info_path.exists():
    train_indices, val_indices, test_indices = gen_split_indices(len(page_path_list), seed=42)
    split_info = {
        "train": [page_path_list[idx].stem for idx in train_indices],
        "validation": [page_path_list[idx].stem for idx in val_indices],
        "test": [page_path_list[idx].stem for idx in test_indices]
    }

    write_json_file(split_info, split_info_path)

else:
    split_info = read_json_file(split_info_path)

train_paths = [path for path in page_path_list if path.stem in split_info["train"]]
val_paths = [path for path in page_path_list if path.stem in split_info["validation"]]

# Create dataset object
train_dataset = create_dset_from_paths(train_paths)
val_dataset = create_dset_from_paths(val_paths)


#%%
# Load model
processor = TrOCRProcessor.from_pretrained(TROCR_PATH)
model = VisionEncoderDecoderModel.from_pretrained(TROCR_PATH).to(device)
model.eval()

#%%

idx = 19

pixel_values = processor(images=train_dataset[idx]["image"], return_tensors="pt").pixel_values

generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(train_dataset[idx]["answer"])
print(generated_text)
# %%
