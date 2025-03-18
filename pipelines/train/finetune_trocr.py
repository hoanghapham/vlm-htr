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
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, get_scheduler

from src.tasks.utils import create_dset_from_paths, create_split_info
from src.tasks.running_text import create_trocr_collate_fn, RunningTextDataset
from src.file_tools import read_json_file
from src.train import Trainer
from src.logger import CustomLogger


#%%
parser = ArgumentParser()
parser.add_argument("--model-name", required=True)
parser.add_argument("--train-epochs", default=10)
parser.add_argument("--batch-size", default=2)
parser.add_argument("--use-data-pct", default=1)
args = parser.parse_args()

# Constants
DATA_DIR        = PROJECT_DIR / "data/polis_line"
MODEL_NAME      = args.model_name
MODEL_OUT_DIR   = PROJECT_DIR / "models" / MODEL_NAME

if not MODEL_OUT_DIR.exists():
    MODEL_OUT_DIR.mkdir(parents=True)

BATCH_SIZE = int(args.batch_size)
USE_DATA_PCT = float(args.use_data_pct)
TRAIN_EPOCHS = int(args.train_epochs)

#%%

logger = CustomLogger(MODEL_NAME, log_to_local=True, log_path=PROJECT_DIR / "logs")
tsb_logger = SummaryWriter(log_dir=PROJECT_DIR / "logs_tensorboard" / MODEL_NAME )

# Load model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REMOTE_MODEL_PATH = "microsoft/trocr-base-handwritten"

logger.info(f"Load model. Use device: {DEVICE}")

processor = TrOCRProcessor.from_pretrained(REMOTE_MODEL_PATH)
model = VisionEncoderDecoderModel.from_pretrained(REMOTE_MODEL_PATH).to(DEVICE)

# Config model
model.config.decoder_start_token_id = processor.tokenizer.eos_token_id
model.config.pad_token_id           = processor.tokenizer.pad_token_id
model.config.vocab_size             = model.config.decoder.vocab_size


# Unfreeze params for training
for param in model.parameters():
    param.is_trainable = True


#%%
# Load data
logger.info("Load data")

# Collect page lists
page_path_list = sorted([path for path in DATA_DIR.glob("*") if path.is_dir()])

split_info_path = DATA_DIR / "split_info.json"

if not split_info_path.exists():
    create_split_info(DATA_DIR, seed=42)

split_info = read_json_file(split_info_path)

train_paths = [path for path in page_path_list if path.stem in split_info["train"]]
val_paths = [path for path in page_path_list if path.stem in split_info["validation"]]

# Create dataset object
train_dataset = create_dset_from_paths(train_paths, RunningTextDataset)
val_dataset = create_dset_from_paths(val_paths, RunningTextDataset)

# Create data loader

collate_fn = create_trocr_collate_fn(processor, DEVICE)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          collate_fn=collate_fn, num_workers=0, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                          collate_fn=collate_fn, num_workers=0)


#%%
# Setup training
torch.manual_seed(42)
MAX_TRAIN_STEPS = int(USE_DATA_PCT * len(train_loader))

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

logger.info(f"Total samples: {len(train_dataset):,}, batch size: {BATCH_SIZE}, total batches: {len(train_loader):,}, max train steps: {MAX_TRAIN_STEPS:,}")
logger.info(f"Start training")

#%%
trainer = Trainer(
    model                = model,
    optimizer            = optimizer,
    lr_scheduler         = lr_scheduler,
    train_loader         = train_loader,
    val_loader           = val_loader,
    n_epochs             = TRAIN_EPOCHS,
    start_epoch          = 1,
    max_train_steps      = MAX_TRAIN_STEPS,
    model_out_dir        = MODEL_OUT_DIR,
    logger               = logger,
    tsb_logger           = tsb_logger,
    load_last_checkpoint = True
)

trainer.train()