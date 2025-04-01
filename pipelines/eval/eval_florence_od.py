#%%
import sys
from pathlib import Path
from argparse import ArgumentParser

import torch
from tqdm import tqdm
from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM, AutoProcessor

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from src.logger import CustomLogger
from src.data_processing.florence import FlorenceTask, FlorenceTextODDataset, predict
from src.data_processing.visual_tasks import bbox_xyxy_to_polygon
from src.train import load_best_checkpoint, load_last_checkpoint
from src.file_tools import write_json_file, write_ndjson_file
from src.evaluation.visual_metrics import precision_recall_fscore, region_coverage
#%%

parser = ArgumentParser()
parser.add_argument("--model-name", required=True)
parser.add_argument("--data-dir", required=True)
parser.add_argument("--checkpoint", default="best", choices=["last", "best", "vanilla"])
parser.add_argument("--object-class", default="region")
parser.add_argument("--debug", required=False, default="false")
args = parser.parse_args()

# args = parser.parse_args([
#     "--model-name", "florence_base__mixed__page__region_od__lora",
#     "--data-dir", "/Users/hoanghapham/Projects/vlm/data/pages/mixed",
#     "--checkpoint", "best",
#     "--mode", "debug"
# ])

# Setup paths
MODEL_NAME          = args.model_name
DATA_DIR            = Path(args.data_dir)
CHECKPOINT          = args.checkpoint
CHECKPOINT_PATH     = args.checkpoint_path
OBJECT_CLASS        = args.object_class
DEBUG               = args.mode == "true"
MAX_ITERS           = 2

LOCAL_MODEL_PATH    = PROJECT_DIR / "models" / MODEL_NAME
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

if "lora" in MODEL_NAME:
    config = LoraConfig.from_pretrained(PROJECT_DIR / "configs/lora")
    model = get_peft_model(model, config)

# Load checkpoint to evaluate

cp_train_metrics = {}

if CHECKPOINT == "vanilla":
    logger.info(f"Evaluate vanilla model: {REMOTE_MODEL_PATH}")
else:
    if CHECKPOINT == "last":
        model, _, cp_train_metrics = load_last_checkpoint(model=model, optimizer=None, model_path=LOCAL_MODEL_PATH, device=DEVICE)
    elif CHECKPOINT == "best":
        model, _, cp_train_metrics = load_best_checkpoint(model=model, optimizer=None, model_path=LOCAL_MODEL_PATH, device=DEVICE, compare_metric="avg_val_loss")
    elif CHECKPOINT == "specific":
        model, _, cp_train_metrics = CHECKPOINT(model=model, optimizer=None, cp_path=CHECKPOINT_PATH, device=DEVICE)

    logger.info(f"Evaluate checkpoint: {cp_train_metrics}")

# Set model to evaluation mode
model.eval()


#%%
# Load test data
test_dataset = FlorenceTextODDataset(DATA_DIR / "test", task=FlorenceTask.OD, object_class="region")
logger.info(f"Total test samples: {len(test_dataset)}")


# Evaluate
task = FlorenceTask.OD

full_results    = []
predictions     = []
annotations     = []
coverage_ratios = []

counter = 0

for data in tqdm(test_dataset, desc="Evaluate"):

    ann = data["original_bboxes"]

    raw_output, parsed_output = predict(
        model, 
        processor, 
        task_prompt=task,
        user_prompt=None, 
        image=data["image"], 
        device=DEVICE
    )

    pred     = parsed_output[task]["bboxes"]
    pred_polygons   = [bbox_xyxy_to_polygon(box) for box in pred]
    ann_polygons    = [bbox_xyxy_to_polygon(box) for box in ann]
    coverage        = region_coverage(pred_polygons, ann_polygons)

    predictions.append(pred)
    annotations.append(ann)
    coverage_ratios.append(coverage)

    full_results.append(
        dict(
            img_name        = Path(data["image_path"]).name,
            ann_bboxes      = ann,
            pred_bboxes     = pred,
            coverage_str    = str(coverage),
            coverage_float  = float(coverage)
        )
    )

    if DEBUG:
        counter += 1
        if counter >= MAX_ITERS:
            break

precision, recall, fscore = precision_recall_fscore(predictions, annotations, iou_threshold=0.5)
avg_region_coverage = float(sum(coverage_ratios))

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
