import sys
from pathlib import Path
from argparse import ArgumentParser

import torch
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

# Parser
parser = ArgumentParser()
parser.add_argument("--data-dir", required=True)
parser.add_argument("--data-fraction", default=1)
parser.add_argument("--base-model-path", required=True, default="yolo11m.pt")
parser.add_argument("--model-name", required=True)
parser.add_argument("--epochs", default=100)
parser.add_argument("--batch-size", default=10)
parser.add_argument("--img-size", default=1280)
parser.add_argument("--resume", default="true")
args = parser.parse_args()

# Basics
DATA_DIR            = Path(args.data_dir)
BASE_MODEL_PATH     = args.base_model_path
MODEL_NAME          = args.model_name
DATASET_CONFIG      = DATA_DIR / "config.yaml"
DEVICE              = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAINED_MODEL_DIR   = PROJECT_DIR / "models" / MODEL_NAME

# Train configs
DATA_FRACTION   = float(args.data_fraction)
EPOCHS          = int(args.epochs)
BATCH_SIZE      = int(args.batch_size)
IMG_SIZE        = int(args.img_size)
RESUME          = args.resume == "true"

MODEL_PATH = BASE_MODEL_PATH

if RESUME:
    TRAINED_MODEL_PATH = TRAINED_MODEL_DIR / "weights/last.pt"
    if TRAINED_MODEL_PATH.exists():
        MODEL_PATH = TRAINED_MODEL_PATH
    else:
        print("Trained model not found, start from scratch")
        RESUME = False

# Init model
model = YOLO(MODEL_PATH)

# Train
training_results = model.train(
    data        = DATASET_CONFIG,
    fraction    = DATA_FRACTION,
    epochs      = EPOCHS,
    batch       = BATCH_SIZE,
    imgsz       = IMG_SIZE,
    device      = DEVICE,
    save        = True,
    save_period = 1,
    project     = str(PROJECT_DIR / "models"),
    name        = MODEL_NAME,
    seed        = 42,
    single_cls  = True,
    resume      = RESUME,
    val         = True,
    plots       = True
)

