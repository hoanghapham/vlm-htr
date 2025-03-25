import sys
from pathlib import Path
from argparse import ArgumentParser

import torch
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

# Parser
parser = ArgumentParser()
parser.add_argument("--data-dir", "-d", required=True)
parser.add_argument("--data-fraction", "-df", default=1)
parser.add_argument("--model-path", "-v", required=True, default="yolo11m.pt")
parser.add_argument("--epochs", "-e", default=100)
parser.add_argument("--batch-size", "-bs", default=10)
parser.add_argument("--img-size", "-is", default=1280)
args = parser.parse_args()

# Basics
DATA_DIR        = Path(args.data_dir)
MODEL_PATH      = args.model_path
MODEL_NAME      = Path(args.model_path).stem
DATASET_CONFIG  = DATA_DIR / "config.yaml"
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train configs
DATA_FRACTION   = float(args.data_fraction)
EPOCHS          = int(args.epochs)
BATCH_SIZE      = int(args.batch_size)
IMG_SIZE        = int(args.img_size)

# Init model
model = YOLO(MODEL_PATH)

# Train
training_results = model.train(
    data        = DATASET_CONFIG,
    fraction    = DATA_FRACTION,
    epochs      = EPOCHS,
    batch_size  = BATCH_SIZE,
    imgsz       = IMG_SIZE,
    device      = DEVICE,
    save        = True,
    save_period = 1,
    project     = PROJECT_DIR / f"models/{MODEL_NAME}",
    name        = f"{MODEL_NAME}_{Path(DATA_DIR).stem}",
    seed        = 42,
    single_cls  = True,
    resume      = True,
    val         = True,
    plots       = True
)

