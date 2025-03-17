#%%
import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).parent.parent
sys.path.append(str(PROJECT_DIR))

from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, get_scheduler

from tqdm import tqdm


from src.htr_tools import create_dset_from_paths, create_trocr_collate_fn
from src.file_tools import read_json_file, write_json_file
from src.utils import gen_split_indices, load_last_checkpoint
from src.logger import CustomLogger


#%%
parser = ArgumentParser()
parser.add_argument("--train-epochs", default=10)
parser.add_argument("--batch-size", default=2)
parser.add_argument("--use-data-pct", default=0.5)
parser.add_argument("--demo", default="false")
args = parser.parse_args()

#%%

logger = CustomLogger("ft_trocr", log_to_local=True, log_path=PROJECT_DIR / "logs")
writer = SummaryWriter(log_dir=PROJECT_DIR / "logs_tensorboard/ft_trocr")

# Load model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REMOTE_MODEL_PATH = "microsoft/trocr-base-handwritten"

logger.info(f"Load model. Use device: {DEVICE}")

processor = TrOCRProcessor.from_pretrained(REMOTE_MODEL_PATH)
model = VisionEncoderDecoderModel.from_pretrained(REMOTE_MODEL_PATH).to(DEVICE)

# Config model
model.config.decoder_start_token_id = processor.tokenizer.eos_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size


# Unfreeze params for training
for param in model.parameters():
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
train_dataset = create_dset_from_paths(train_paths)
val_dataset = create_dset_from_paths(val_paths)

# Create data loader
BATCH_SIZE = int(args.batch_size)

collate_fn = create_trocr_collate_fn(processor, DEVICE)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          collate_fn=collate_fn, num_workers=0, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                          collate_fn=collate_fn, num_workers=0)

logger.info(f"Total samples: {len(train_dataset)}, batch size: {BATCH_SIZE}, total batches: {len(train_loader)}")


#%%
# Setup training

MODEL_DIR = PROJECT_DIR / "models/trocr-htr-line"

if not MODEL_DIR.exists():
    MODEL_DIR.mkdir(parents=True)


torch.manual_seed(42)
TRAIN_EPOCHS = int(args.train_epochs)
START_EPOCH = 1
USE_NBATCH = int(float(args.use_data_pct) * len(train_loader))

if args.demo == "true":
    USE_NBATCH = 2

# Set optimizer & scheduler
# Scheduler & optimizer config: https://github.com/microsoft/unilm/tree/master/trocr#fine-tuning-on-iam

optimizer = AdamW(
    model.parameters(), 
    lr=2e-5, 
    weight_decay=0.0001,
)

lr_scheduler = get_scheduler(
    name="inverse_sqrt", 
    optimizer=optimizer,
    num_warmup_steps=0, 
    num_training_steps=TRAIN_EPOCHS * len(train_loader),
    
    # inverse_sqrt params. See https://github.com/facebookresearch/fairseq/blob/main/fairseq/optim/lr_scheduler/inverse_squa
    scheduler_specific_kwargs=dict(
        warmup_init_lr=1e-8,
        warmup_updates=500
    )
)
# Load state
logger.info("Find last checkpoint")
last_checkpoint = load_last_checkpoint(MODEL_DIR, DEVICE)

if last_checkpoint is not None:
    model.load_state_dict(last_checkpoint["model_state_dict"])
    optimizer = AdamW(model.parameters(), lr=1e-6)
    optimizer.load_state_dict(last_checkpoint["optimizer_state_dict"])
    last_loss = last_checkpoint["loss"]
    START_EPOCH = last_checkpoint["epoch"] + 1
    logger.info(f"Last epoch: {START_EPOCH}, loss: {last_loss}")

# Train

for epoch in range(START_EPOCH, TRAIN_EPOCHS + 1):
    torch.cuda.empty_cache()

    model.train()
    train_loss = 0
    
    # Inputs is the processed tuple (text, image)
    iterator = tqdm(train_loader, desc=f"Train epoch {epoch}/{TRAIN_EPOCHS}", total=USE_NBATCH)

    for batch_idx, batch_data in enumerate(iterator):
        
        # Skip a portion of the batches
        if batch_idx > USE_NBATCH:
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

    write_json_file(checkpoint_metrics, MODEL_DIR / f"checkpoint_epoch_{epoch:04d}.json")
    torch.save(checkpoint_states, MODEL_DIR / f"checkpoint_epoch_{epoch:04d}.pt")

    avg_train_loss = train_loss / len(train_loader)
    logger.info(f"Average Training Loss: {avg_train_loss:.4f}")
    writer.add_scalar("Train loss", avg_train_loss, epoch)

    
    # Check validation loss
    model.eval()
    val_loss = 0
    with torch.no_grad():
        iterator = tqdm(val_loader, desc=f"Validate epoch {epoch}/{TRAIN_EPOCHS}")
        for batch_idx, batch_data in enumerate(iterator):
            
            if args.demo == "true":
                if batch_idx > USE_NBATCH:
                    break

            outputs = model(**batch_data)
            loss = outputs.loss
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    logger.info(f"Average Validation Loss: {avg_val_loss:.4f}")
    writer.add_scalar("Val loss", avg_val_loss, epoch)
#%%