#%%
from pathlib import Path
from argparse import ArgumentParser

import torch
from PIL import Image

from vlm.utils.file_tools import list_files
from vlm.utils.logger import CustomLogger
from vlm.data_processing.visual_tasks import IMAGE_EXTENSIONS
from vlm.evaluation.utils import evaluate_one_page, evaluate_multiple_pages
from vlm.htr.pipelines.florence import FlorencePipeline

# Setup
parser = ArgumentParser()
parser.add_argument("--split-type", required=True, default="mixed", choices=["mixed", "sbs"])
parser.add_argument("--checkpoint", default="best")
parser.add_argument("--batch-size", default=6)
parser.add_argument("--device", default="cuda", choices="cpu")
parser.add_argument("--debug", required=False, default="false")
args = parser.parse_args()

# args = parser.parse_args([
#     "--split-type", "mixed",
#     "--batch-size", "2",
#     "--device", "cpu",
#     "--debug", "true"
# ])


SPLIT_TYPE      = args.split_type
CHECKPOINT      = args.checkpoint
BATCH_SIZE      = int(args.batch_size)
DEBUG           = args.debug == "true"
PROJECT_DIR     = Path(__file__).parent.parent.parent
TEST_DATA_DIR   = PROJECT_DIR / f"data/page/{SPLIT_TYPE}/test/"
OUTPUT_DIR      = PROJECT_DIR / f"evaluations/pipeline_florence__{SPLIT_TYPE}__single_model" / CHECKPOINT

img_paths = list_files(TEST_DATA_DIR, IMAGE_EXTENSIONS)
xml_paths = list_files(TEST_DATA_DIR, [".xml"])

if DEBUG:
    img_paths = [img_paths[202]]
    xml_paths = [xml_paths[202]]

#%%

logger = CustomLogger(f"pl_florence__{SPLIT_TYPE}__single_model")

# Load models
if args.device == "cuda":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = args.device


# Load two instances of the same model
pipeline = FlorencePipeline(
    pipeline_type       = "line_od__ocr",
    line_od_model_path  = PROJECT_DIR / f"models/trained/florence_base__{SPLIT_TYPE}__line_od__ocr" / CHECKPOINT,
    ocr_model_path      = PROJECT_DIR / f"models/trained/florence_base__{SPLIT_TYPE}__line_od__ocr" / CHECKPOINT,
    batch_size          = BATCH_SIZE,
    device              = DEVICE,
    logger              = logger
)

#%%
pipeline_outputs = []

for img_idx, (img_path, xml_path) in enumerate(zip(img_paths, xml_paths)):

    # Skip if the file is already processed
    img_metric_path = OUTPUT_DIR / (Path(img_path).stem + "__metrics.json")
    if img_metric_path.exists() and not DEBUG:
        logger.info(f"Skip: {img_path.name}")
        continue

    logger.info(f"Image {img_idx}/{len(img_paths)}: {img_path.name}")
    image       = Image.open(img_path).convert("RGB")

    ## Run pipeline
    page_output = pipeline.run(image)
    pipeline_outputs.append(page_output)

    # Evaluate
    page_metrics = evaluate_one_page(page_output, xml_path, OUTPUT_DIR)
    logger.info(f"Metrics: {page_metrics.float_str}")


# %%
evaluate_multiple_pages(pipeline_outputs, xml_paths, OUTPUT_DIR)
