#%%
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoProcessor, get_scheduler
from datasets import load_from_disk, concatenate_datasets
from tqdm import tqdm
from argparse import ArgumentParser
import json
import typing

import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).parent.parent
sys.path.append(str(PROJECT_DIR))

from src.logger import CustomLogger
from src.utils import gen_split_indices


# Setup arg parse
parser = ArgumentParser()
parser.add_argument("--epochs", default=1)
parser.add_argument("--use-batch-pct", default=0.5)
parser.add_argument("--batch-size", default=2)
args = parser.parse_args([])

# Vars
EPOCHS = int(args.epochs)
USE_BATCH_PCT = float(args.use_batch_pct)
BATCH_SIZE = int(args.batch_size)
DATA_DIR = PROJECT_DIR / "data/poliskammare_region"
MODEL_DIR = PROJECT_DIR / "models/florence-2-base-ft-htr-region"

if not MODEL_DIR.exists():
    MODEL_DIR.mkdir(parents=True)

#%%

# Helpers
class HTRDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        question = "<SwedishHTR>Print out the text in this image"
        answer = example["transcription"]
        image = example['image'].convert("RGB")
        return question, answer, image
    
    def select(self, indices: typing.Iterable):
        subset = [self.data[int(idx)] for idx in indices]
        return HTRDataset(subset)


def create_dataset(path_list: list[str | Path]) -> HTRDataset:
    dsets = []
    for path in path_list:
        dsets.append(load_from_disk(path))
    data = concatenate_datasets(dsets)
    return HTRDataset(data)


# Create train loader & validate loader

def create_collate_fn(processor):
    def func(batch):
        questions, answers, images = zip(*batch)
        inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(device)
        labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True, return_token_type_ids=False).input_ids.to(device)
        return dict(
            input_ids=inputs["input_ids"], 
            pixel_values=inputs["pixel_values"], 
            labels=labels
        )

    return func


#%%

logger = CustomLogger("ft_florence_htr_region")

# Load model
logger.info("Load model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "microsoft/Florence-2-base-ft"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    revision='refs/pr/6'
).to(device)

# Unfreeze vision params
for param in model.vision_tower.parameters():
    param.is_trainable = True


processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True, revision='refs/pr/6'
)

collate_fn = create_collate_fn(processor)

#%%
# Load data
logger.info("Load data")

# Collect page lists
page_list = [str(path) for path in DATA_DIR.glob("*") if path.is_dir()]

train_indices, val_indices, test_indices = gen_split_indices(len(page_list), seed=42)
split_info = {
    "train": [page_list[idx] for idx in train_indices],
    "validation": [page_list[idx] for idx in val_indices],
    "test": [page_list[idx] for idx in test_indices]
}

with open(DATA_DIR / "split_info.json", "w") as f:
    json.dump(split_info, f)


train_pages = [page_list[idx] for idx in train_indices]
val_pages = [page_list[idx] for idx in val_indices]

train_dataset = create_dataset(train_pages)
val_dataset = create_dataset(val_pages)


# Create data loader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=0, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=0)


#%%
# Train

optimizer = AdamW(model.parameters(), lr=1e-6)
num_training_steps = EPOCHS * len(train_loader)
lr_scheduler = get_scheduler(name="linear", optimizer=optimizer,
                              num_warmup_steps=0, num_training_steps=num_training_steps,)

BREAK_IDX = int(USE_BATCH_PCT * len(train_loader))


#%%
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    # Inputs is the processed tuple (text, image)
    iterator = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{EPOCHS}")

    for batch_idx, batch_data in enumerate(iterator):
        
        # Only use a certain percentage of the batch
        if batch_idx > BREAK_IDX:
            break

        # Predict output
        outputs = model(**batch_data)

        # Calculate loss, then backward
        loss = outputs.loss
        loss.backward()

        # Then step
        optimizer.step()
        lr_scheduler.step()

        # Reset grad
        optimizer.zero_grad()
        train_loss += loss.item()

        iterator.set_postfix({"loss": loss.item()})
    
    # Save checkpoint
    model_info = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(model_info, MODEL_DIR / f"checkpoint_epoch_{epoch:03d}.pt")

    avg_train_loss = train_loss / len(train_loader)
    logger.info(f"Average Training Loss: {avg_train_loss}")


#%%
# Check validation loss
model.eval()
val_loss = 0
with torch.no_grad():
    for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"):
        outputs = model(**batch)
        loss = outputs.loss
        val_loss += loss.item()


avg_val_loss = val_loss / len(val_loader)
logger.info(f"Average Validation Loss: , {avg_val_loss}")
