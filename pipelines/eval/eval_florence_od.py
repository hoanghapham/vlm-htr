#%%
import sys
from pathlib import Path
from argparse import ArgumentParser

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from src.logger import CustomLogger
from src.data_processing.florence import FlorenceTask, FlorencePageTextODDataset, predict
from src.data_processing.visual_tasks import bbox_xyxy_to_polygon
from src.train import load_checkpoint
from src.file_tools import write_json_file, write_ndjson_file
from src.evaluation.visual_metrics import compute_bbox_precision_recall_fscore, compute_polygons_region_coverage
#%%

parser = ArgumentParser()
parser.add_argument("--model-name", required=True)
parser.add_argument("--data-dir", required=True)
parser.add_argument("--checkpoint", default="best", choices=["last", "best", "vanilla"])
parser.add_argument("--checkpoint-path", required=False)
parser.add_argument("--object-class", default="region")
parser.add_argument("--batch-size", default=2)
parser.add_argument("--debug", required=False, default="false")
args = parser.parse_args()

# args = parser.parse_args([
#     "--model-name", "florence_base__mixed__page__region_od__lora",
#     "--data-dir", "/Users/hoanghapham/Projects/vlm/data/page/mixed",
#     "--checkpoint", "best",
#     "--mode", "debug"
# ])

# Setup paths
MODEL_NAME          = args.model_name
DATA_DIR            = Path(args.data_dir)
CHECKPOINT          = args.checkpoint
CHECKPOINT_PATH     = args.checkpoint_path
OBJECT_CLASS        = args.object_class
BATCH_SIZE          = int(args.batch_size)
DEBUG               = args.debug == "true"
MAX_ITERS           = 2

LOCAL_MODEL_PATH    = PROJECT_DIR / "models/trained" / MODEL_NAME
OUTPUT_DIR          = PROJECT_DIR / "evaluations" / MODEL_NAME
REMOTE_MODEL_PATH   = "microsoft/Florence-2-base-ft"
REVISION            = 'refs/pr/6'
DEVICE              = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True)

# Logger
logger = CustomLogger(f"eval__{MODEL_NAME}", log_to_local=True)

#%%
# Load model
logger.info("Load model")
processor   = AutoProcessor.from_pretrained(REMOTE_MODEL_PATH, trust_remote_code=True, device_map=DEVICE)
model       = AutoModelForCausalLM.from_pretrained(REMOTE_MODEL_PATH, trust_remote_code=True, device_map=DEVICE)

# Load checkpoint to evaluate

cp_train_metrics = {}

if CHECKPOINT == "vanilla":
    logger.info(f"Evaluate vanilla model: {REMOTE_MODEL_PATH}")
else:
    if CHECKPOINT in ["last", "best"]:
        cp_path = LOCAL_MODEL_PATH / CHECKPOINT
    elif CHECKPOINT == "specific":
        cp_path = CHECKPOINT_PATH

    model, _, _, cp_train_metrics = load_checkpoint(model=model, cp_path=LOCAL_MODEL_PATH / CHECKPOINT, device=DEVICE)
    logger.info(f"Evaluate checkpoint: {cp_train_metrics}")


# Set model to evaluation mode
model.eval()


#%%
# Load test data
test_dataset = FlorencePageTextODDataset(DATA_DIR, task=FlorenceTask.OD, object_class=OBJECT_CLASS)


# Evaluate
task = FlorenceTask.OD

full_results    = []
predictions     = []
groundtruths     = []
coverages = []

counter = 0

iterator = list(range(0, len(test_dataset), BATCH_SIZE))

logger.info(f"Total test samples: {len(test_dataset)}, batches: {len(iterator)}")

for start_idx in tqdm(iterator, desc="Evaluate"):

    batch_data = test_dataset[start_idx:start_idx+BATCH_SIZE]
    images = [data["image"] for data in batch_data]

    _, parsed_output = predict(
        model, 
        processor, 
        task_prompt=task,
        user_prompt=None, 
        images=images, 
        device=DEVICE
    )

    if parsed_output == []:
        logger.info(f"No results for batch {start_idx/BATCH_SIZE}")
        continue

    for in_data, out_data in zip(batch_data, parsed_output):

        pred_bboxes     = out_data[task]["bboxes"]
        pred_polygons   = [bbox_xyxy_to_polygon(box) for box in pred_bboxes]

        gt_bboxes       = in_data["original_bboxes"]
        gt_polygons     = [bbox_xyxy_to_polygon(box) for box in gt_bboxes]
        coverage        = compute_polygons_region_coverage(pred_polygons, gt_polygons)

        predictions += pred_bboxes
        groundtruths += gt_bboxes
        coverages.append(coverage)

        full_results.append(
            dict(
                img_name        = Path(in_data["image_path"]).name,
                gt_bboxes       = gt_bboxes,
                pred_bboxes     = pred_bboxes,
                coverage_str    = str(coverage),
                coverage_float  = float(coverage)
            )
        )

    if DEBUG:
        counter += 1
        if counter >= MAX_ITERS:
            break


precision, recall, fscore = compute_bbox_precision_recall_fscore(predictions, groundtruths, iou_threshold=0.5)
avg_region_coverage = float(sum(coverages))

logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, Fscore: {fscore:.4f}, Avg. region coverage: {avg_region_coverage:.4f}")


#%%
metrics = dict(
    precision=precision,
    recall=recall,
    fscore=fscore,
    avg_region_coverage=avg_region_coverage
)

metrics.update(cp_train_metrics)

step_idx_str = str(cp_train_metrics["step_idx"]).zfill(10)

write_ndjson_file(full_results, OUTPUT_DIR / f"full_results_step_{step_idx_str}.json")
write_json_file(metrics, OUTPUT_DIR / f"metrics_{step_idx_str}.json")
    
logger.info(f"Wrote results to {OUTPUT_DIR}")
