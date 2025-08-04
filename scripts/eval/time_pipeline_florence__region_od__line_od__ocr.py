#%%
from pathlib import Path
from argparse import ArgumentParser

import torch
import time
from PIL import Image

from vlm.utils.file_tools import list_files
from vlm.utils.logger import CustomLogger
from vlm.data_processing.visual_tasks import IMAGE_EXTENSIONS
from vlm.htr.pipelines.florence import FlorencePipeline
from vlm.evaluation.utils import evaluate_one_page


# Setup
parser = ArgumentParser()
parser.add_argument("--split-type", required=True, default="mixed", choices=["mixed", "sbs"])
parser.add_argument("--batch-size", default=6)
parser.add_argument("--device", default="cuda", choices="cpu")
args = parser.parse_args()

# args = parser.parse_args([
#     "--split-type", "mixed",
#     "--batch-size", "2",
#     "--device", "cpu",
#     "--debug", "true",
# ])

SPLIT_TYPE      = args.split_type
BATCH_SIZE      = int(args.batch_size)
PROJECT_DIR     = Path(__file__).parent.parent.parent
TEST_DATA_DIR   = PROJECT_DIR / f"data/page/{SPLIT_TYPE}/test/"
OUTPUT_DIR      = PROJECT_DIR / f"evaluations/pipeline_florence__{SPLIT_TYPE}__region_od__line_od__ocr"

img_paths = list_files(TEST_DATA_DIR, IMAGE_EXTENSIONS)[:100]
xml_paths = list_files(TEST_DATA_DIR, [".xml"])[:100]

#%%

logger = CustomLogger(f"time_pl_flor__{SPLIT_TYPE}__3steps")

# Load models
if args.device == "cuda":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = args.device

pipeline = FlorencePipeline(
    pipeline_type           = "region_od__line_od__ocr",
    region_od_model_path    = PROJECT_DIR / f"models/trained/florence_base__{SPLIT_TYPE}__page__region_od/best",
    line_od_model_path      = PROJECT_DIR / f"models/trained/florence_base__{SPLIT_TYPE}__region__line_od/best",
    ocr_model_path          = PROJECT_DIR / f"models/trained/florence_base__{SPLIT_TYPE}__line_bbox__ocr/best",
    batch_size              = BATCH_SIZE,
    device                  = DEVICE,
    logger                  = logger
)


#%%
pipeline_outputs = []
all_times = []


# Iterate through images
for img_idx, (img_path, xml_path) in enumerate(zip(img_paths, xml_paths)):

    # Skip if the file is already processed
    img_metric_path = OUTPUT_DIR / (Path(img_path).stem + "__metrics.json")

    logger.info(f"Image {img_idx}/{len(img_paths)}: {img_path.name}")
    image       = Image.open(img_path).convert("RGB")

    # Run pipeline
    t0 = time.time()
    page_output = pipeline.run(image)
    t1 = time.time()
    all_times.append(t1-t0)

    pipeline_outputs.append(page_output)

    # Evaluate
    page_metrics = evaluate_one_page(page_output, xml_path)
    logger.info(f"Metrics: {page_metrics.float_str}")

#%%
# Average across all images:
avg_time = sum(all_times) / len(all_times)
logger.info(f"Total time: {sum(all_times) / 60:.2f}, avg time: {avg_time / 60:.2f}")