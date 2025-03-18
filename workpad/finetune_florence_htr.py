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
from transformers import AutoModelForCausalLM, AutoProcessor, get_scheduler

from src.file_tools import read_json_file
from src.tasks.utils import create_dset_from_paths, create_split_info
from src.tasks.running_text import RunningTextDataset, create_florence_collate_fn
from src.train import Trainer
from src.logger import CustomLogger


#%%
parser = ArgumentParser()
parser.add_argument("--data-dir", required=True)
parser.add_argument("--model-name", required=True)
parser.add_argument("--train-epochs", default=5)
parser.add_argument("--batch-size", default=2)
parser.add_argument("--use-data-pct", default=0.5)
args = parser.parse_args()

# Setup constant values
MODEL_NAME      = args.model_name
BATCH_SIZE      = int(args.batch_size)
TRAIN_EPOCHS    = int(args.train_epochs)
DATA_DIR        = Path(args.data_dir)
USE_DATA_PCT    = float(args.use_data_pct)
MODEL_OUT_DIR   = PROJECT_DIR / "models" / MODEL_NAME

if not MODEL_OUT_DIR.exists():
    MODEL_OUT_DIR.mkdir(parents=True)

# Setup loggers
logger = CustomLogger(MODEL_NAME, log_to_local=True)
tsb_logger = SummaryWriter(log_dir = PROJECT_DIR / f"logs_tensorboard/{MODEL_NAME}")

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

# Collect page lists
local_path_list = sorted([path for path in DATA_DIR.glob("*") if path.is_dir()])

# Subset train & validate set
split_info_path = DATA_DIR / "split_info.json"

if not split_info_path.exists():
    create_split_info(DATA_DIR, seed=42)

split_info = read_json_file(split_info_path)
train_paths = [path for path in local_path_list if path.stem in split_info["train"]]
val_paths = [path for path in local_path_list if path.stem in split_info["validation"]]

# Create dataset object
train_dataset = create_dset_from_paths(train_paths, RunningTextDataset)
val_dataset = create_dset_from_paths(val_paths, RunningTextDataset)

# Create data loader

collate_fn = create_florence_collate_fn(processor, DEVICE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          collate_fn=collate_fn, num_workers=0, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                          collate_fn=collate_fn, num_workers=0)


#%%
# Setup training

START_EPOCH = 1
MAX_TRAIN_STEPS = int(USE_DATA_PCT * len(train_loader))
TOTAL_TRAIN_STEPS = TRAIN_EPOCHS * len(train_loader)

optimizer = AdamW(model.parameters(), lr=1e-6)
lr_scheduler = get_scheduler(
    name="linear", 
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=TOTAL_TRAIN_STEPS
)

# Load state
#%%
# Train
logger.info(f"Total samples: {len(train_dataset):,}, batch size: {BATCH_SIZE}, total batches: {len(train_loader):,}, max train steps: {MAX_TRAIN_STEPS:,}")
logger.info(f"Start training")

trainer = Trainer(
    model                = model,
    optimizer            = optimizer,
    lr_scheduler         = lr_scheduler,
    train_loader         = train_loader,
    val_loader           = val_loader,
    n_epochs             = TRAIN_EPOCHS,
    start_epoch          = START_EPOCH,
    max_train_steps      = MAX_TRAIN_STEPS,
    model_out_dir        = MODEL_OUT_DIR,
    logger               = logger,
    tsb_logger           = tsb_logger,
    load_last_checkpoint = True
)

trainer.train()
