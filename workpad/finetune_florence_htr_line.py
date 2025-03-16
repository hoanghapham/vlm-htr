#%%
import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).parent.parent
sys.path.append(str(PROJECT_DIR))

from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoProcessor, get_scheduler
from tqdm import tqdm

from src.htr_tools import create_dset_from_paths, create_collate_fn
from src.file_tools import read_json_file, write_json_file
from src.utils import gen_split_indices, load_last_checkpoint
from src.logger import CustomLogger


#%%
parser = ArgumentParser()
parser.add_argument("--train-epochs", default=5)
parser.add_argument("--batch-size", default=2)
parser.add_argument("--use-data-pct", default=0.5)
args = parser.parse_args()

logger = CustomLogger("ft_florence_htr_line", log_to_local=True)

# Load model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REMOTE_MODEL_PATH = "microsoft/Florence-2-base-ft"

logger.info(f"Load model. Use device: {DEVICE}")

model = AutoModelForCausalLM.from_pretrained(
    REMOTE_MODEL_PATH,
    trust_remote_code=True,
    revision='refs/pr/6'
).to(DEVICE)

processor = AutoProcessor.from_pretrained(
    REMOTE_MODEL_PATH,
    trust_remote_code=True, revision='refs/pr/6'
)

# Unfreeze vision params
for param in model.vision_tower.parameters():
    param.is_trainable = True


#%%
# Load data
logger.info("Load data")
DATA_DIR = PROJECT_DIR / "data/polis_line"

# Collect page lists
page_path_list = sorted([path for path in DATA_DIR.glob("*") if path.is_dir()])

# Create dataset
# Subset train & validate set

split_info_path = DATA_DIR / "split_info.json"

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
train_dataset = create_dset_from_paths(train_paths).select(range(2))
val_dataset = create_dset_from_paths(val_paths).select(range(2))

# Create data loader
BATCH_SIZE = int(args.batch_size)
collate_fn = create_collate_fn(processor, DEVICE)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          collate_fn=collate_fn, num_workers=0, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                          collate_fn=collate_fn, num_workers=0)


#%%
logger.info("Start training")

MODEL_DIR = PROJECT_DIR / "models/florence-2-base-ft-htr-line"

if not MODEL_DIR.exists():
    MODEL_DIR.mkdir(parents=True)

TRAIN_EPOCHS = int(args.train_epochs)
START_EPOCH = 0
BREAK_IDX = int(float(args.use_data_pct) * len(train_loader))
num_training_steps = TRAIN_EPOCHS * len(train_loader)

optimizer = AdamW(model.parameters(), lr=1e-6)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer,
    num_warmup_steps=0, num_training_steps=num_training_steps,
)

# Load state
last_checkpoint = load_last_checkpoint(MODEL_DIR, DEVICE)

if last_checkpoint is not None:
    model.load_state_dict(last_checkpoint["model_state_dict"])
    optimizer = AdamW(model.parameters(), lr=1e-6)
    optimizer.load_state_dict(last_checkpoint["optimizer_state_dict"])
    START_EPOCH = last_checkpoint["epoch"]
    last_loss = last_checkpoint["loss"]
    logger.info(f"Last epoch: {START_EPOCH + 1}, loss: {last_loss}")

#%%
# Train
for epoch in range(START_EPOCH, TRAIN_EPOCHS):
    logger.info(f"Epoch {epoch + 1}/{TRAIN_EPOCHS}")

    model.train()
    train_loss = 0
    
    # Inputs is the processed tuple (text, image)
    iterator = tqdm(train_loader, desc="Train", total=BREAK_IDX, unit="batch")

    for batch_idx, batch_data in enumerate(iterator):
        
        # Skip half of the batches
        if batch_idx > BREAK_IDX:
            break

        # Predict output
        outputs = model(**batch_data)

        # Calculate loss, then backward
        loss = outputs.loss
        loss.backward()

        # Then step to update weights
        optimizer.step()
        lr_scheduler.step()

        # Reset grad
        optimizer.zero_grad()
        train_loss += loss.item()

        iterator.set_postfix({"loss": loss.item()})
    
    # Save checkpoint
    checkpoint_metrics = {
        'epoch': epoch,
        'loss': loss.item(),
    }

    checkpoint_states = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    write_json_file(checkpoint_metrics, MODEL_DIR / f"checkpoint_epoch_{epoch:03d}.json")
    torch.save(checkpoint_states, MODEL_DIR / f"checkpoint_epoch_{epoch:03d}.pt")

    avg_train_loss = train_loss / len(train_loader)
    logger.info(f"Average Training Loss: {avg_train_loss}")


    # Check validation loss
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{TRAIN_EPOCHS}"):
            outputs = model(**batch)
            loss = outputs.loss
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    logger.info(f"Average Validation Loss: , {avg_val_loss}")
