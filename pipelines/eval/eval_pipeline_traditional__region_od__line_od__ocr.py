#%%
import sys
from pathlib import Path
from argparse import ArgumentParser

import torch
from PIL import Image

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from src.file_tools import list_files
from src.data_processing.visual_tasks import IMAGE_EXTENSIONS
from src.logger import CustomLogger
from src.htr.pipelines.traditional import TraditionalPipeline
from src.htr.pipelines.evaluation import evaluate_pipeline


# Setup
parser = ArgumentParser()
parser.add_argument("--split-type", required=True, default="mixed", choices=["mixed", "sbs"])
parser.add_argument("--batch-size", default=6)
parser.add_argument("--debug", required=False, default="false")
# args = parser.parse_args()

args = parser.parse_args([
    "--split-type", "mixed",
    "--batch-size", "6",
    "--debug", "true"
])

SPLIT_TYPE      = args.split_type
BATCH_SIZE      = int(args.batch_size)
TEST_DATA_DIR   = PROJECT_DIR / f"data/page/{SPLIT_TYPE}/test/"
OUTPUT_DIR      = PROJECT_DIR / f"evaluations/pipeline_traditional__{SPLIT_TYPE}__region_od__line_od__ocr"
DEBUG           = args.debug == "true"

img_paths = list_files(TEST_DATA_DIR, IMAGE_EXTENSIONS)
xml_paths = list_files(TEST_DATA_DIR, [".xml"])

if DEBUG:
    img_paths = [img_paths[0], img_paths[704]]
    xml_paths = [xml_paths[0], xml_paths[704]]
    OUTPUT_DIR = OUTPUT_DIR / "debug"

#%%

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger  = CustomLogger(f"pl_trad__{SPLIT_TYPE}__3steps")

# Prepare pipeline
pipeline = TraditionalPipeline(
    pipeline_type           = "region_od__line_od__ocr",
    region_od_model_path    = PROJECT_DIR / f"models/trained/yolo11m__{SPLIT_TYPE}__page__region_od/weights/best.pt",
    line_od_model_path     = PROJECT_DIR / f"models/trained/yolo11m_seg__{SPLIT_TYPE}__region__line_od/weights/best.pt",
    ocr_model_path          = PROJECT_DIR / f"models/trained/trocr_base__{SPLIT_TYPE}__line_od__ocr/best",
    batch_size              = BATCH_SIZE,
    device                  = DEVICE,
    logger                  = logger
)



#%%
pipeline_outputs = []

for img_idx, (img_path, xml_path) in enumerate(zip(img_paths, xml_paths)):
    # logger.info(f"Processing image {img_idx+1}/{len(img_paths)}")

    # Skip if the file is already processed
    img_metric_path = OUTPUT_DIR / (Path(img_path).stem + "__metrics.json")
    if img_metric_path.exists() and not DEBUG:
        logger.info(f"Skip: {img_path.name}")
        continue

    logger.info(f"Image {img_idx}/{len(img_paths)}: {img_path.name}")
    image = Image.open(img_path).convert("RGB")

    ## Run pipeline
    page_output = pipeline.region_od__line_od__ocr(image)
    pipeline_outputs.append(page_output)
    

#%%
# Evaluate:
evaluate_pipeline(pipeline_outputs, xml_paths, OUTPUT_DIR)