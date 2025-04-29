#%%
import sys
from pathlib import Path
from argparse import ArgumentParser

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor
import pandas as pd

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from src.logger import CustomLogger
from src.data_processing.florence import FlorenceTask, FlorenceSingleLineSegDataset, predict

from src.train import load_checkpoint
from src.file_tools import write_json_file, write_ndjson_file
from src.evaluation.visual_metrics import polygon_to_mask, compute_seg_metrics
#%%

parser = ArgumentParser()
parser.add_argument("--model-name", required=True)
parser.add_argument("--data-dir", required=True)
parser.add_argument("--checkpoint", default="best", choices=["last", "best", "vanilla"])
parser.add_argument("--batch-size", default=2)
parser.add_argument("--debug", required=False, default="false")
args = parser.parse_args()

# args = parser.parse_args([
#     "--model-name", "florence_base__mixed__line_cropped__line_seg",
#     "--data-dir", "/Users/hoanghapham/Projects/vlm/data/page/mixed",
#     "--checkpoint", "best",
#     # "--mode", "debug"
# ])

# Setup paths
MODEL_NAME          = args.model_name
DATA_DIR            = Path(args.data_dir)
CHECKPOINT          = args.checkpoint
BATCH_SIZE          = int(args.batch_size)
DEBUG               = args.debug == "true"

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

    model, _, _, cp_train_metrics = load_checkpoint(model=model, cp_path=LOCAL_MODEL_PATH / CHECKPOINT, device=DEVICE)
    logger.info(f"Evaluate checkpoint: {cp_train_metrics}")


# Set model to evaluation mode
model.eval()


#%%
# Load test data
logger.info("Load test data")
test_dataset = FlorenceSingleLineSegDataset(DATA_DIR)
logger.info(f"Total test samples: {len(test_dataset)}")

#%%
# Evaluate
task = FlorenceTask.REGION_TO_SEGMENTATION

full_results    = []
predictions     = []
groundtruths    = []
all_metrics     = []

logger.info("Evaluate")

iterator = list(range(0, len(test_dataset), BATCH_SIZE))

for start_idx in tqdm(iterator):
    batch = test_dataset[start_idx:start_idx+BATCH_SIZE]
    images = [data["image"] for data in batch]

    _, parsed_output = predict(
        model, 
        processor, 
        task_prompt=task,
        user_prompt=None, 
        images=images, 
        device=DEVICE
    )

    pred_masks_florence = [output[FlorenceTask.REGION_TO_SEGMENTATION]["polygons"][0][0] for output in parsed_output]

    for i in range(BATCH_SIZE):
        try:
            pred_mask_num       = list(zip(pred_masks_florence[i][::2], pred_masks_florence[i][1::2]))
            pred_mask_binary    = polygon_to_mask(pred_mask_num, images[i].size)
            gt_mask_num         = batch[i]["polygon"]
            gt_mask_binary      = polygon_to_mask(gt_mask_num, images[i].size)
            metrics             = compute_seg_metrics(pred_mask_binary, gt_mask_binary)

            all_metrics.append(metrics)

            full_results.append(
                dict(
                    unique_key = batch[i]["unique_key"],
                    gt_mask_num = gt_mask_num,
                    pred_mask_num = pred_mask_num,
                    metrics = metrics
                )
            )

        except Exception as e:
            logger.exception(e)

# Write full result
write_ndjson_file(full_results, OUTPUT_DIR / "full_results.json")


# Calculate average metrics
avg_metrics = pd.DataFrame(all_metrics).mean().to_dict()
avg_metrics_str = ", ".join([f"{k}: \t {round(v, 4)}" for k, v in avg_metrics.items()])
logger.info(f"Avg metrics: {avg_metrics_str}")
write_json_file(avg_metrics, OUTPUT_DIR / "avg_metrics.json")

logger.info(f"Wrote results to {OUTPUT_DIR}")