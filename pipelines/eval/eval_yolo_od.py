#%%
import sys
import numpy as np
from pathlib import Path
from argparse import ArgumentParser

import torch
from ultralytics import YOLO
from tqdm import tqdm

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from src.file_tools import list_files, write_ndjson_file, write_json_file
from src.data_processing.visual_tasks import Bbox
from src.data_processing.utils import XMLParser
from src.evaluation.visual_metrics import precision_recall_fscore, region_coverage
from src.logger import CustomLogger
#%%

parser = ArgumentParser()
parser.add_argument("--data-dir", required=True)
parser.add_argument("--model-name", required=True)
parser.add_argument("--batch-size", default=100)
parser.add_argument("--object-class", required=True, default="region")
args = parser.parse_args()

DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR       = Path(args.data_dir)
MODEL_NAME      = args.model_name
OBJECT_CLASS    = args.object_class
BATCH_SIZE      = int(args.batch_size)
MODEL_PATH      = PROJECT_DIR / f"models/{MODEL_NAME}/weights/best.pt"
OUTPUT_DIR      = PROJECT_DIR / "evaluations" / MODEL_NAME

if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True)

logger = CustomLogger(f"eval__{MODEL_NAME}", log_to_local=False)

#%%
img_paths = list_files(DATA_DIR / "images", [".tif", ".jpg"])
xml_paths = list_files(DATA_DIR / "page_xmls", [".xml"])
matched = set([path.stem for path in img_paths]).intersection(set([path.stem for path in xml_paths]))

assert len(img_paths) == len(xml_paths) == len(matched) > 0, \
    f"Length invalid, or mismatch img-xml pairs: {len(img_paths)} images, {len(xml_paths)} XML files, {len(matched)} matches"

# Get annotations
xml_parser = XMLParser()
annotations = []

logger.info("Get annotations")
for path in xml_paths:
    if OBJECT_CLASS == "region":
        objects = xml_parser.get_regions(path)
    elif OBJECT_CLASS == "line":
        objects = xml_parser.get_lines(path)
    img_ann_bboxes = [Bbox(obj["bbox"]) for obj in objects]
    annotations.append(img_ann_bboxes)


# %%
logger.info("Get predictions")
model = YOLO(MODEL_PATH)

results = []
iterator = list(range(0, len(img_paths), BATCH_SIZE))

for i in tqdm(iterator, total=len(iterator), unit="batch"):
    batch = img_paths[i:i+BATCH_SIZE]
    batch_results = model.predict(batch, verbose=False, device=DEVICE)
    results += batch_results


#%%
predictions = []

for result in results:
    img_pred_bboxes = []
    for box_tensor in result.boxes.xyxy:
        xyxy = [data.item() for data in box_tensor]
        img_pred_bboxes.append(Bbox(xyxy))
    
    predictions.append(img_pred_bboxes)

assert len(annotations) == len(predictions), f"predictions & annotations length mismatched"
# %%

precision, recall, fscore = precision_recall_fscore(predictions, annotations)
logger.info(f"Precision: {precision}, Recall: {recall}, Fscore: {fscore}")

page_region_coverages = []
for pred, ann in zip(predictions, annotations):
    pred_polygons = [box.polygon for box in pred]
    ann_polygons = [box.polygon for box in ann]
    coverage = region_coverage(pred_polygons, ann_polygons)
    page_region_coverages.append(coverage)

avg_region_coverage = float(sum(page_region_coverages)) / len(page_region_coverages)
logger.info(f"Average region coverage: {avg_region_coverage:.4f}")

# Write metrics
metrics = dict(
    precision=precision,
    recall=recall,
    fscore=fscore,
    avg_region_coverage=avg_region_coverage
)

write_json_file(metrics, OUTPUT_DIR / "metrics.json")

# Write results
all_results = []

for img_path, ann, pred, coverage in zip(img_paths, annotations, predictions, page_region_coverages):
    all_results.append(
        dict(
            img_path = str(img_path),
            ann_bboxes = ann,
            pred_bbox = pred,
            coverage = coverage
        )
    )

write_ndjson_file(all_results, OUTPUT_DIR / "predictions.json")

logger.info(f"Wrote results to {OUTPUT_DIR}")