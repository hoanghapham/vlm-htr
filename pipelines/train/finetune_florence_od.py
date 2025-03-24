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
from peft import LoraConfig, get_peft_model

from src.file_tools import read_json_file
from src.data_process.utils import normalize_name
from src.data_process.florence import FlorenceTextODDataset, create_florence_collate_fn
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
parser.add_argument("--use-lora", default="false")
parser.add_argument("--detect-class", default="region")
args = parser.parse_args()

# args = parser.parse_args([
#     "--data-dir", "/Users/hoanghapham/Projects/thesis-data/riksarkivet",
#     "--model-name", "demo",
#     "--num-train-epochs", "5",
#     "--max-train-steps", "11",
#     "--batch-size", "2",
#     "--logging-interval", "3"
# ])


# Setup constant values
MODEL_NAME          = args.model_name
BATCH_SIZE          = int(args.batch_size)
NUM_TRAIN_EPOCHS    = int(args.num_train_epochs)
MAX_TRAIN_STEPS     = int(args.max_train_steps)
LOGGING_INTERVAL    = int(args.logging_interval)
DATA_DIR            = Path(args.data_dir)
MODEL_OUT_DIR       = PROJECT_DIR / "models" / MODEL_NAME
USE_LORA            = args.use_lora == "true"
DETECT_CLASS        = args.detect_class

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
    trust_remote_code=True, revision=REVISION
)

# Unfreeze all params
for param in model.parameters():
    param.is_trainable = True


#%%
# Load data
logger.info("Load data")

# Collect page lists

img_paths = sorted(DATA_DIR.glob(pattern="images/**/*.tif"))
xml_paths = sorted(DATA_DIR.glob(pattern="page_xmls/**/*.xml"))

split_info = read_json_file(DATA_DIR / "split_info.json")
# train_names = [normalize_name(name) for name in split_info["train"]]
# val_names = [normalize_name(name) for name in split_info["validation"]]

# train_dataset   = FlorenceTextODDataset(
#     img_paths=[path for path in img_paths if normalize_name(path.stem) in train_names],
#     xml_paths=[path for path in xml_paths if normalize_name(path.stem) in train_names]
# )
# val_dataset   = FlorenceTextODDataset(
#     img_paths=[path for path in img_paths if normalize_name(path.stem) in val_names],
#     xml_paths=[path for path in xml_paths if normalize_name(path.stem) in val_names]
# )

train_dataset   = FlorenceTextODDataset(split_info["train"], object_class=DETECT_CLASS)
val_dataset     = FlorenceTextODDataset(split_info["validation"], object_class=DETECT_CLASS)

# Create data loader

collate_fn = create_florence_collate_fn(processor, DEVICE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          collate_fn=collate_fn, num_workers=0, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                          collate_fn=collate_fn, num_workers=0)

logger.info(f"Total samples: {len(train_dataset):,}. Batch size: {BATCH_SIZE}. Total batches: {len(train_loader):,}. Max train steps: {MAX_TRAIN_STEPS:,}")
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

# LORA

TARGET_MODULES = [
    "q_proj", "o_proj", "k_proj", "v_proj", 
    "linear", "Conv2d", "lm_head", "fc2"
]

config = LoraConfig(
    r=8,
    lora_alpha=8,
    target_modules=TARGET_MODULES,
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
    bias="none",
    inference_mode=False,
    use_rslora=True,
    init_lora_weights="gaussian",
    revision=REVISION
)


# Load state
#%%
# Train

logger.info(f"Start training")


if USE_LORA:
    peft_model = get_peft_model(model, config)
    trainable_params, all_param = peft_model.get_nb_trainable_parameters()

    logger.info(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )

    trainer = Trainer(
        model                = peft_model,
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

else:
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
