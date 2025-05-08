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
from src.htr.pipelines.florence import FlorencePipeline
from src.htr.pipelines.evaluation import evaluate_pipeline


# Setup
parser = ArgumentParser()
parser.add_argument("--split-type", required=True, default="mixed", choices=["mixed", "sbs"])
parser.add_argument("--batch-size", default=6)
parser.add_argument("--device", default="cuda", choices="cpu")
parser.add_argument("--debug", required=False, default="false")
args = parser.parse_args()

# args = parser.parse_args([
#     "--split-type", "sbs",
#     "--batch-size", "2",
#     "--device", "cpu",
#     "--debug", "true"
# ])

SPLIT_TYPE      = args.split_type
BATCH_SIZE      = int(args.batch_size)
DEBUG           = args.debug == "true"
TEST_DATA_DIR   = PROJECT_DIR / f"data/page/{SPLIT_TYPE}/test/"
OUTPUT_DIR      = PROJECT_DIR / f"evaluations/pipeline_florence__{SPLIT_TYPE}__line_od__ocr"

img_paths = list_files(TEST_DATA_DIR, IMAGE_EXTENSIONS)
xml_paths = list_files(TEST_DATA_DIR, [".xml"])

if DEBUG:
    img_paths = [img_paths[0], img_paths[136]]
    xml_paths = [xml_paths[0], xml_paths[136]]
    OUTPUT_DIR = OUTPUT_DIR / "debug"

#%%

logger = CustomLogger(f"pl_flor__{SPLIT_TYPE}__2steps")

# Load models
if args.device == "cuda":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = args.device

pipeline = FlorencePipeline(
    pipeline_type="line_od__ocr",
    line_od_model_path  = PROJECT_DIR / f"models/trained/florence_base__{SPLIT_TYPE}__page__line_od/best",
    ocr_model_path      = PROJECT_DIR / f"models/trained/florence_base__{SPLIT_TYPE}__line_bbox__ocr/best",
    batch_size          = BATCH_SIZE,
    device              = DEVICE,
    logger              = logger
)

#%%
pipeline_outputs = []

# Iterate through pages
for img_idx, (img_path, xml_path) in enumerate(zip(img_paths, xml_paths)):

    # Skip if the file is already processed
    img_metric_path = OUTPUT_DIR / (Path(img_path).stem + "__metrics.json")
    if img_metric_path.exists() and not DEBUG:
        logger.info(f"Skip: {img_path.name}")
        continue

    logger.info(f"Image {img_idx}/{len(img_paths)}: {img_path.name}")
    image = Image.open(img_path).convert("RGB")

    ## Run pipeline
    page_output = pipeline.run(image, sort_mode="consider_margins")
    pipeline_outputs.append(page_output)


evaluate_pipeline(pipeline_outputs, xml_paths, OUTPUT_DIR)

# %%

# from src.visualization import draw_bboxes_xyxy
# idx = 1
# draw_bboxes_xyxy(images[idx], results[idx].bboxes)
# gt_lines = xml_parser.get_lines(xml_path=xml_paths[idx])
# gt_regions = xml_parser.get_regions(xml_path=xml_paths[idx])
# draw_bboxes_xyxy(images[idx], [data["bbox"] for data in gt_regions])
# draw_bboxes_xyxy(images[idx], [data["bbox"] for data in gt_lines])

#%%




