# Train Florence-2 for the OCR task
#%%
import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoProcessor, get_scheduler

from src.data_processing.florence import FlorenceOCRDataset, create_collate_fn
from src.vlm.train import Trainer
from src.logger import CustomLogger


#%%
parser = ArgumentParser()
parser.add_argument("--data-dir", required=True)
parser.add_argument("--model-name", required=True)
parser.add_argument("--num-train-epochs", default=5)
parser.add_argument("--max-train-steps", default=2000)
parser.add_argument("--logging-interval", default=100)
parser.add_argument("--batch-size", default=2)
parser.add_argument("--user-prompt", required=False)
parser.add_argument("--debug", default=False)
args = parser.parse_args()

# args = parser.parse_args([
#     "--data-dir", str(PROJECT_DIR / "data/line_seg/mixed"),
#     "--model-name", "demo",
#     "--num-train-epochs", "1",
#     "--max-train-steps", "1",
#     "--batch-size", "2",
#     "--logging-interval", "1",
#     "--debug", "true"
# ])


# Setup constant values
MODEL_NAME          = args.model_name
BATCH_SIZE          = int(args.batch_size)
NUM_TRAIN_EPOCHS    = int(args.num_train_epochs)
MAX_TRAIN_STEPS     = int(args.max_train_steps)
LOGGING_INTERVAL    = int(args.logging_interval)
DATA_DIR            = Path(args.data_dir)
USER_PROMPT         = args.user_prompt  # Can be used as the custom question
DEBUG               = args.debug == "true"

MODEL_OUT_DIR       = PROJECT_DIR / "models" / MODEL_NAME

if not MODEL_OUT_DIR.exists():
    MODEL_OUT_DIR.mkdir(parents=True)

# Setup loggers
logger = CustomLogger(MODEL_NAME, log_to_local=True)
tsb_logger = SummaryWriter(log_dir = PROJECT_DIR / "logs_tensorboard" / MODEL_NAME)

#%%
# Load model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REMOTE_MODEL_PATH = "microsoft/Florence-2-base-ft"
REVISION = 'refs/pr/6'

logger.info(f"Load model. Use device: {DEVICE}")

model = AutoModelForCausalLM.from_pretrained(
    REMOTE_MODEL_PATH,
    trust_remote_code=True,
    revision=REVISION
).to(DEVICE)

processor = AutoProcessor.from_pretrained(
    REMOTE_MODEL_PATH,
    trust_remote_code=True, 
    revision=REVISION
)

# All params are unfrozen by default
# for param in model.parameters():
#     param.requires_grad = True


#%%
# Load data
logger.info("Load data")

# custom_question can be None
train_dataset   = FlorenceOCRDataset(DATA_DIR / "train", custom_question=USER_PROMPT)
val_dataset     = FlorenceOCRDataset(DATA_DIR / "val", custom_question=USER_PROMPT)

# Create data loader
collate_fn      = create_collate_fn(processor, DEVICE)
train_loader    = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
val_loader      = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

logger.info(f"Total train samples: {len(train_dataset):,}, batch size: {BATCH_SIZE}, total batches: {len(train_loader):,}, max train steps: {MAX_TRAIN_STEPS:,}")


#%%
# Setup training
TOTAL_TRAIN_STEPS = NUM_TRAIN_EPOCHS * len(train_loader)

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

trainer = Trainer(
    model                = model,
    optimizer            = optimizer,
    lr_scheduler         = lr_scheduler,
    train_loader         = train_loader,
    val_loader           = val_loader,
    num_train_epochs     = NUM_TRAIN_EPOCHS,
    max_train_steps      = MAX_TRAIN_STEPS,
    resume               = True,
    model_out_dir        = MODEL_OUT_DIR,
    logger               = logger,
    tsb_logger           = tsb_logger,
    logging_interval     = LOGGING_INTERVAL,
    debug                = DEBUG
)


trainer.train()

# %%
