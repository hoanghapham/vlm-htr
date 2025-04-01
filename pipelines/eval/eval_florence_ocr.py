#%%
import sys
from pathlib import Path
from argparse import ArgumentParser

import torch
from tqdm import tqdm
from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM, AutoProcessor
from htrflow.evaluate import CER, WER, BagOfWords

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from src.logger import CustomLogger
from src.data_processing.florence import FlorenceTask, FlorenceOCRDataset, predict
from src.train import load_best_checkpoint, load_last_checkpoint, load_checkpoint
from src.file_tools import write_json_file, write_list_to_text_file
#%%

parser = ArgumentParser()
parser.add_argument("--model-name", required=True)
parser.add_argument("--data-dir", required=True)
parser.add_argument("--checkpoint", default="best", choices=["last", "best", "vanilla", "specific"])
parser.add_argument("--checkpoint-path", required=False)
parser.add_argument("--user-prompt", required=False)
parser.add_argument("--debug", required=False, default="false")
args = parser.parse_args()

# args = parser.parse_args([
#     "--model-name", "florence_base__mixed__line__ocr",
#     "--data-dir", "/Users/hoanghapham/Projects/vlm/data/lines/mixed",
#     "--mode", "debug"
# ])

# Setup paths
MODEL_NAME          = args.model_name
DATA_DIR            = Path(args.data_dir)
CHECKPOINT          = args.CHECKPOINT
CHECKPOINT_PATH     = Path(args.checkpoint_path)
USER_PROMPT         = args.user_prompt
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
logger.info("Load test data")
test_dataset = FlorenceOCRDataset(DATA_DIR, custom_question=USER_PROMPT)

logger.info(f"Total test samples: {len(test_dataset)}")
logger.info(f"User prompt: {USER_PROMPT}")

# Evaluate
#%%
task = FlorenceTask.OCR
cer = CER()
wer = WER()
bow = BagOfWords()

cer_list = []
wer_list = []
bow_hits_list = []
bow_extras_list = []
gt_list = []
pred_list = []

counter = 0

# Can only process one image at a time due to post_process_generation task requiring image size,
# but image size varies
for data in tqdm(test_dataset, desc="Evaluate"):

    groundtruth = data["answer"]
    raw_output, parsed_output = predict(
        model, 
        processor, 
        task_prompt=task,
        user_prompt=USER_PROMPT, 
        image=data["image"], 
        device=DEVICE
    )

    # Calcualte metrics
    cer_value = cer.compute(parsed_output[task], groundtruth)["cer"]
    wer_value = wer.compute(parsed_output[task], groundtruth)["wer"]
    bow_hits_value = bow.compute(parsed_output[task], groundtruth)["bow_hits"]
    bow_extras_value = bow.compute(parsed_output[task], groundtruth)["bow_extras"]

    # Append results
    cer_list.append(cer_value)
    wer_list.append(wer_value)
    bow_hits_list.append(bow_hits_value)
    bow_extras_list.append(bow_extras_value)
    gt_list.append(groundtruth)
    pred_list.append(parsed_output)

    if DEBUG:
        counter += 1
        if counter >= MAX_ITERS:
            break


#%%
avg_cer = float(sum(cer_list))
avg_wer = float(sum(wer_list))
avg_bow_hits = float(sum(bow_hits_list))
avg_bow_extras = float(sum(bow_extras_list))

logger.info(f"Avg. CER: {avg_cer:.4f}, Avg. WER: {avg_wer:.4f}")
logger.info(f"Avg. BoW hits: {avg_bow_hits:.4f}, Avg. BoW extras: {avg_bow_extras:.4f}")


#%%

# Save results
# Avg metrics

logger.info(f"Save result to {OUTPUT_DIR}")

step_idx_str = str(cp_train_metrics["step_idx"]).zfill(10)

metrics_aggr = {
    "step_idx": cp_train_metrics["step_idx"],
    "avg_train_loss": cp_train_metrics["avg_train_loss"],
    "avg_val_loss": cp_train_metrics["avg_val_loss"],
    "cer": avg_cer,
    "wer": avg_wer,
    "bow_hits": avg_bow_hits,
    "bow_extras": avg_bow_extras
}

write_json_file(metrics_aggr, OUTPUT_DIR / f"metrics_aggr_step_{step_idx_str}.json")

# Detailed results
metrics_lists = {
    "cer": [str(val) for val in cer_list],
    "wer": [str(val) for val in wer_list],
    "bow_hits": [str(val) for val in bow_hits_list],
    "bow_extras": [str(val) for val in bow_extras_list]
}

write_json_file(metrics_lists, OUTPUT_DIR / f"metrics_lists_step_{step_idx_str}.json")

# Write ground text for reference
write_list_to_text_file(gt_list, OUTPUT_DIR / "ground_truth.txt")

# Write prediction for reference
pred_list = [pred[task] for pred in pred_list]
write_list_to_text_file(pred_list, OUTPUT_DIR / f"prediction_step_{step_idx_str}.txt")

