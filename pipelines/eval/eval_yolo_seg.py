#%%
import sys
from pathlib import Path
from argparse import ArgumentParser

import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from src.file_tools import list_files, write_json_file, read_lines, write_list_to_text_file
from src.data_processing.visual_tasks import yolo_seg_to_coords, sort_polygons, IMAGE_EXTENSIONS
from src.evaluation.visual_metrics import match_and_evaluate
from src.logger import CustomLogger
#%%

parser = ArgumentParser()
parser.add_argument("--data-dir", required=True)
parser.add_argument("--model-name", required=True)
parser.add_argument("--checkpoint", default="best")
parser.add_argument("--batch-size", default=10)
parser.add_argument("--device", default="cuda")
args = parser.parse_args()

# args = parser.parse_args([
#     "--data-dir", str(PROJECT_DIR / "data/yolo/mixed/inst_seg_lines_within_regions/test/"),
#     "--model-name", "yolo11m_seg__mixed__region__line_seg",
#     "--checkpoint", "best",
#     "--batch-size", "2",
# ])


DEVICE              = torch.device(args.device)
DATA_DIR            = Path(args.data_dir)
MODEL_NAME          = args.model_name
BATCH_SIZE          = int(args.batch_size)
CHECKPOINT          = args.checkpoint
MODEL_PATH          = PROJECT_DIR / "models/yolo_base/yolo11m.pt" if CHECKPOINT == "vanilla" else \
                        PROJECT_DIR / f"models/{MODEL_NAME}/weights/{CHECKPOINT}.pt"
OUTPUT_DIR          = PROJECT_DIR / "evaluations" / MODEL_NAME

if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True)

logger = CustomLogger(f"eval__{MODEL_NAME}", log_to_local=False)

#%%
img_paths = list_files(DATA_DIR / "images", IMAGE_EXTENSIONS)
label_paths = list_files(DATA_DIR / "labels", [".txt"])
matched = set([path.stem for path in img_paths]).intersection(set([path.stem for path in label_paths]))

assert len(img_paths) == len(label_paths) == len(matched) > 0, \
    f"Length invalid, or mismatch img-label pairs: {len(img_paths)} images, {len(label_paths)} txt files, {len(matched)} matches"


# Get annotations
annotations = []
valids = []
invalids = []

logger.info("Get annotations")
for idx, (img_path, label_path) in enumerate(zip(img_paths, label_paths)):
    image = Image.open(img_path)
    label_lines = read_lines(label_path)
    gt_polygons = [np.array(yolo_seg_to_coords(line, image.width, image.height)[1]) for line in label_lines]
    
    try:
        gt_polygons = sort_polygons(gt_polygons)
        annotations.append(gt_polygons)
        valids.append(idx)
    except Exception as e:
        print(e)
        invalids.append(idx)

logger.info(f"Total images: {len(img_paths)}, valid annotations: {len(valids)}")

invalid_files = [str(label_paths[idx]) for idx in invalids]
write_list_to_text_file(invalid_files, OUTPUT_DIR / "invalid_annotations.txt")


# %%
logger.info("Get predictions")
model = YOLO(MODEL_PATH)

processed_indices = []
predictions = []
no_pred = []
iterator = list(range(0, len(valids), BATCH_SIZE))

for i in tqdm(iterator, total=len(iterator), unit="batch"):
    selected = valids[i:i+BATCH_SIZE]
    batch = [img_paths[idx] for idx in selected]

    # Get prediction
    try:
        batch_results = model.predict(batch, verbose=False, device=DEVICE)
        processed_indices += selected
    except torch.OutOfMemoryError:
        print("OutOfMemoryError")

    # Get seg masks from YOLO's prediction
    for idx, result in enumerate(batch_results):
        try:
            pred_polygons = result.masks.xy
            pred_polygons = sort_polygons(pred_polygons)
            predictions.append(pred_polygons)
        except Exception as e:
            print(e)
            no_pred.append(batch[idx])

logger.info(f"Processed {len(processed_indices)}/{len(valids)}")    
filtered_ann = []
for idx in processed_indices:
    try:
        filtered_ann.append(annotations[idx])
    except Exception as e:
        print(e)


# Save annotations & Predictions incase we want to do manual analysis
torch.save(filtered_ann, OUTPUT_DIR / "annotations.pt")
torch.save(predictions, OUTPUT_DIR / "predictions.pt")
write_list_to_text_file(no_pred, OUTPUT_DIR / "no_prediction.txt")

logger.info(f"Predictions: {len(predictions)}, filtered annotations: {len(filtered_ann)}")
# %%

page_metrics = []
line_metrics = []

for idx, (pred, gt) in tqdm(enumerate(zip(predictions, annotations)), total = len(predictions)):
    try:
        image = Image.open(img_paths[idx])
        page_result = match_and_evaluate(pred, gt, image.size, iou_threshold=0.5)
        filename = img_paths[idx].stem

        page_metric = {
            "filename": filename,
            "num_preds": page_result["num_preds"], 
            "num_gts": page_result["num_gts"],
            "unmatched_preds": page_result["unmatched_preds"], 
            "unmatched_gts": page_result["unmatched_gts"], 
        }
        page_metrics.append(page_metric)

        for line in page_result["matched"]:
            line_metric = {
                "filename": filename,
                "pair": line["pair"],
                **line["metrics"]
            }
            line_metrics.append(line_metric)
    except Exception as e:
        logger.exception(e)


# Write full results
page_metrics_df = pd.DataFrame(page_metrics)
page_metrics_df.to_csv(OUTPUT_DIR / "page_metrics.csv", index=False)

#%%
# Write line-based results
line_metrics_df = pd.DataFrame(line_metrics)
line_metrics_df.to_csv(OUTPUT_DIR / "line_metrics.csv", index=False)

#%%

# Average resuls
columns = [
    "iou",
    "dice",
    "pixel_accuracy",
    "mean_pixel_accuracy",
    "boundary_f1",
    "region_coverage",
]

avg_metrics = line_metrics_df[columns].mean().to_dict()
write_json_file(avg_metrics, OUTPUT_DIR / "avg_metrics.json")

avg_metrics_str = "\n".join([f"{k}: \t {round(v, 4)}" for k, v in avg_metrics.items()])
logger.info(f"Avg metrics: \n{avg_metrics_str}")
logger.info(f"Wrote results to {OUTPUT_DIR}")
# %%
