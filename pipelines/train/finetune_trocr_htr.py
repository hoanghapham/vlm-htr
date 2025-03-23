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

from src.data_process.trocr import TrOCRLineDataset
from src.file_tools import read_json_file
from src.train import Trainer
from src.logger import CustomLogger


#%%
parser = ArgumentParser()
parser.add_argument("--data-dir", required=True)
parser.add_argument("--model-name", required=True)
parser.add_argument("--num-train-epochs", default=5)
parser.add_argument("--max-train-steps", default=2000)
parser.add_argument("--logging-interval", default=100)
parser.add_argument("--batch-size", default=2)
args = parser.parse_args()

# Constants
MODEL_NAME          = args.model_name
BATCH_SIZE          = int(args.batch_size)
NUM_TRAIN_EPOCHS    = int(args.num_train_epochs)
MAX_TRAIN_STEPS     = int(args.max_train_steps)
LOGGING_INTERVAL    = int(args.logging_interval)
DATA_DIR            = Path(args.data_dir)
MODEL_OUT_DIR       = PROJECT_DIR / "models" / MODEL_NAME

if not MODEL_OUT_DIR.exists():
    MODEL_OUT_DIR.mkdir(parents=True)

logger = CustomLogger(MODEL_NAME, log_to_local=True, log_path=PROJECT_DIR / "logs")
tsb_logger = SummaryWriter(log_dir=PROJECT_DIR / "logs_tensorboard" / MODEL_NAME)


#%%
# Load model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REMOTE_MODEL_PATH = "microsoft/trocr-base-handwritten"

logger.info(f"Load model. Use device: {DEVICE}")

processor = TrOCRProcessor.from_pretrained(REMOTE_MODEL_PATH)
model = VisionEncoderDecoderModel.from_pretrained(REMOTE_MODEL_PATH).to(DEVICE)
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
split_info = read_json_file(DATA_DIR / "split_info.json")

# Create dataset object
train_dataset   = TrOCRLineDataset(split_info["train"])
val_dataset     = TrOCRLineDataset(split_info["validation"])

# Create data loader

def create_collate_fn(processor, device):
    def func(batch):
        images = [data[1]["image"] for data in batch]
        texts = [data[1]["transcription"] for data in batch]
        
        pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device)
        labels = processor.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)["input_ids"].to(device)

        return dict(
            pixel_values=pixel_values, 
            labels=labels,
        )

    return func


collate_fn = create_collate_fn(processor, DEVICE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          collate_fn=collate_fn, num_workers=0, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                          collate_fn=collate_fn, num_workers=0)

logger.info(f"Total samples: {len(train_dataset):,}, batch size: {BATCH_SIZE}, total batches: {len(train_loader):,}")
#%%
# Setup training
torch.manual_seed(42)

# Set optimizer & scheduler
# Scheduler & optimizer config: https://github.com/microsoft/unilm/tree/master/trocr#fine-tuning-on-iam

optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.0001)

lr_scheduler = get_scheduler(
    name="inverse_sqrt", 
    optimizer=optimizer,
    num_warmup_steps=0, 
    num_training_steps=NUM_TRAIN_EPOCHS * len(train_loader),
    
    # inverse_sqrt params. See https://github.com/facebookresearch/fairseq/blob/main/fairseq/optim/lr_scheduler/inverse_squa
    scheduler_specific_kwargs=dict(
        warmup_init_lr=1e-8,
        warmup_updates=500
    )
)

logger.info(f"Start training")

#%%
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
    logging_interval     = LOGGING_INTERVAL
)
trainer.train()