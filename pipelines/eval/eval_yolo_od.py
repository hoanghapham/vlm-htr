#%%
import sys
from pathlib import Path
from argparse import ArgumentParser

import torch
from ultralytics import YOLO
from tqdm import tqdm

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from src.file_tools import list_files, write_ndjson_file, write_json_file
from src.data_processing.visual_tasks import bbox_xyxy_to_polygon, IMAGE_EXTENSIONS
from src.data_processing.utils import XMLParser
from src.evaluation.visual_metrics import precision_recall_fscore, region_coverage
from src.logger import CustomLogger
#%%

parser = ArgumentParser()
parser.add_argument("--data-dir", required=True)
parser.add_argument("--model-name", required=True)
parser.add_argument("--checkpoint", default="best")
parser.add_argument("--batch-size", default=10)
parser.add_argument("--object-class", required=True, default="region")
args = parser.parse_args()

DEVICE              = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR            = Path(args.data_dir)
MODEL_NAME          = args.model_name
OBJECT_CLASS        = args.object_class
BATCH_SIZE          = int(args.batch_size)
CHECKPOINT          = args.checkpoint
MODEL_PATH          = PROJECT_DIR / "models/yolo_base/yolo11m.pt" if CHECKPOINT == "vanilla" else \
                        PROJECT_DIR / f"models/{MODEL_NAME}/weights/{CHECKPOINT}.pt"
OUTPUT_DIR          = PROJECT_DIR / "evaluations" / MODEL_NAME

if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True)

logger = CustomLogger(f"eval__{MODEL_NAME}", log_to_local=False)

#%%
img_paths = list_files(DATA_DIR, IMAGE_EXTENSIONS)
xml_paths = list_files(DATA_DIR, [".xml"])
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
    img_ann_bboxes = [obj["bbox"] for obj in objects]
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
        box = [data.item() for data in box_tensor]
        img_pred_bboxes.append(box)
    
    predictions.append(img_pred_bboxes)

assert len(annotations) == len(predictions), f"predictions & annotations length mismatched"
# %%

precision, recall, fscore = precision_recall_fscore(predictions, annotations)
logger.info(f"Precision: {precision}, Recall: {recall}, Fscore: {fscore}")

coverage_ratios = []
for pred, ann in zip(predictions, annotations):
    pred_polygons = [bbox_xyxy_to_polygon(box) for box in pred]
    ann_polygons = [bbox_xyxy_to_polygon(box) for box in ann]
    coverage = region_coverage(pred_polygons, ann_polygons)
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