#%%
import sys
from pathlib import Path
from argparse import ArgumentParser

import torch
from ultralytics import YOLO
from tqdm import tqdm

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from src.file_tools import write_ndjson_file, write_json_file
from src.data_processing.visual_tasks import bbox_xyxy_to_polygon
from src.data_processing.yolo import YOLOPageLineODDataset, YOLOPageRegionODDataset, YOLORegionLineODDataset
from src.evaluation.visual_metrics import compute_bbox_precision_recall_fscore, compute_polygons_region_coverage
from src.logger import CustomLogger
#%%

parser = ArgumentParser()
parser.add_argument("--data-dir", required=True)
parser.add_argument("--model-name", required=True)
parser.add_argument("--checkpoint", default="best")
parser.add_argument("--batch-size", default=10)
parser.add_argument("--task", required=True, choices=["page__region_od", "page__line_od", "region__line_od"])
parser.add_argument("--debug", required=False, default="false")
args = parser.parse_args()

# args = parser.parse_args([
#     "--data-dir", str(PROJECT_DIR / "data/page/mixed/test"),
#     "--model-name", "yolo11m__mixed__page__region_od",
#     "--checkpoint", "best",
#     "--batch-size", "2",
#     "--task", "page__region_od",
#     "--debug", "true"
# ])

DEVICE              = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR            = Path(args.data_dir)
MODEL_NAME          = args.model_name
TASK                = args.task
BATCH_SIZE          = int(args.batch_size)
CHECKPOINT          = args.checkpoint
DEBUG               = args.debug == "true"
OUTPUT_DIR          = PROJECT_DIR / "evaluations" / MODEL_NAME

if CHECKPOINT == "vanilla":
    MODEL_PATH = PROJECT_DIR / "models/yolo_base/yolo11m.pt"
else:
    MODEL_PATH = PROJECT_DIR / f"models/trained/{MODEL_NAME}/weights/{CHECKPOINT}.pt"


if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True)

logger = CustomLogger(f"eval__{MODEL_NAME}", log_to_local=False)

#%%

if TASK == "page__region_od":
    test_data = YOLOPageRegionODDataset(DATA_DIR)
elif TASK == "page__line_od":
    test_data = YOLOPageLineODDataset(DATA_DIR)
elif TASK == "region__line_od":
    test_data = YOLORegionLineODDataset(DATA_DIR)
else:
    raise NotImplementedError(f"Unknown task: {TASK}")

#%%
if DEBUG:
    test_data = test_data[0:2]

logger.info("Get annotations")
annotations = [data["bboxes"] for data in test_data]

# %%
logger.info("Get predictions")
model = YOLO(MODEL_PATH)

results = []
iterator = list(range(0, len(test_data), BATCH_SIZE))

for i in tqdm(iterator, total=len(iterator), unit="batch"):
    batch_imgs = [data["image"] for data in test_data[i:i+BATCH_SIZE]]
    batch_results = model.predict(batch_imgs, verbose=False, device=DEVICE)
    results += batch_results

#%%
predictions = []

for result in results:
    img_pred_bboxes = []
    for box_tensor in result.boxes.xyxy:
        box = [data.item() for data in box_tensor]
        img_pred_bboxes.append(box)
    
    predictions.append(img_pred_bboxes)

assert len(annotations) == len(predictions), f"predictions & annotations length mismatched"
# %%

precision, recall, fscore = compute_bbox_precision_recall_fscore(predictions, annotations)
logger.info(f"Precision: {precision}, Recall: {recall}, Fscore: {fscore}")

coverage_ratios = []
for pred, ann in zip(predictions, annotations):
    pred_polygons = [bbox_xyxy_to_polygon(box) for box in pred]
    ann_polygons = [bbox_xyxy_to_polygon(box) for box in ann]
    coverage = compute_polygons_region_coverage(pred_polygons, ann_polygons)
    coverage_ratios.append(coverage)

avg_region_coverage = float(sum(coverage_ratios))
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
full_results = []
img_paths = [data["img_path"] for data in test_data]

for img_path, ann, pred, coverage in zip(img_paths, annotations, predictions, coverage_ratios):
    full_results.append(
        dict(
            img_path = str(img_path),
            ann_bboxes = ann,
            pred_bboxes = pred,
            coverage_str = str(coverage),
            coverage_float = float(coverage)
        )
    )

write_ndjson_file(full_results, OUTPUT_DIR / "full_results.json")

logger.info(f"Wrote results to {OUTPUT_DIR}")


# %%
