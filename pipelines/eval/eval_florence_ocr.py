#%%
import sys
from pathlib import Path
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor
from htrflow.evaluate import CER, WER, BagOfWords
from argparse import ArgumentParser

PROJECT_DIR = Path.cwd().parent
sys.path.append(str(PROJECT_DIR))

from src.logger import CustomLogger
from src.data_processing.florence import FlorenceTask, FlorenceOCRDataset, predict

from src.train import load_best_checkpoint, load_last_checkpoint, Checkpoint
from src.file_tools import write_json_file, write_list_to_text_file
#%%

parser = ArgumentParser()
parser.add_argument("--model-name", required=True)
parser.add_argument("--input-dir", required=True)
parser.add_argument("--load-checkpoint", default="best", choices=["last", "best", "vanilla"])
parser.add_argument("--user-prompt", required=False)
args = parser.parse_args()

# args = parser.parse_args([
#     "--model-name", "florence_base__ft_htr_line",
#     "--test-data-dir", str(PROJECT_DIR / "data/hovratt_line"),
# ])

# Setup paths
DEVICE              = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME          = args.model_name
LOCAL_MODEL_PATH    = PROJECT_DIR / "models" / MODEL_NAME
REMOTE_MODEL_PATH   = "microsoft/Florence-2-base-ft"

INPUT_DIR           = Path(args.input_dir)
LOAD_CHECKPOINT     = args.load_checkpoint
USER_PROMPT         = args.user_prompt
OUTPUT_DIR          = PROJECT_DIR / "evaluations" / MODEL_NAME

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

eval_cp = Checkpoint()

if LOAD_CHECKPOINT == "vanilla":
    logger.info(f"Evaluate vanilla model: {REMOTE_MODEL_PATH}")
else:
    if LOAD_CHECKPOINT == "last":
        eval_cp = load_last_checkpoint(LOCAL_MODEL_PATH, DEVICE)
    elif LOAD_CHECKPOINT == "best":
        eval_cp = load_best_checkpoint(LOCAL_MODEL_PATH, "avg_val_loss", DEVICE)

    model.load_state_dict(eval_cp.model_state_dict)
    logger.info(f"Evaluate checkpoint: {eval_cp}")

# Set model to evaluation mode
model.eval()


#%%
# Load test data
logger.info("Load test data")
test_dataset = FlorenceOCRDataset(INPUT_DIR, custo  m_question=USER_PROMPT)

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

# Can only process one image at a time due to post_process_generation task requiring image size,
# but image size varies
for data in tqdm(test_dataset, desc="Evaluate"):

    groundtruth = data["answer"]
    raw_output, parsed_output = predict(
        model, 
        processor, 
        task=task,
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


#%%
avg_cer = float(sum(cer_list))
avg_wer = float(sum(wer_list))
avg_bow_hits = float(sum(bow_hits_list))
avg_bow_extras = float(sum(bow_extras_list))

logger.info(f"Avg. CER: {avg_cer:.4f}, Avg. WER: {avg_wer:.4f}")
logger.info(f"Avg. BoW hits: {avg_bow_hits:.4f}, Avg. BoW extras: {avg_bow_extras:.4f}")


#%%
avg_cer = float(sum(cer_list))
avg_wer = float(sum(wer_list))
avg_bow_hits = float(sum(bow_hits_list))
avg_bow_extras = float(sum(bow_extras_list))

logger.info(f"Avg. CER: {avg_cer:.4f}, Avg. WER: {avg_wer:.4f}")
logger.info(f"Avg. BoW hits: {avg_bow_hits:.4f}, Avg. BoW extras: {avg_bow_extras:.4f}")


# Save results
# Avg metrics

logger.info(f"Save result to {OUTPUT_DIR}")

metrics_aggr = {
    "step_idx": eval_cp.step_idx,
    "avg_train_loss": eval_cp.avg_train_loss,
    "avg_val_loss": eval_cp.avg_val_loss,
    "cer": avg_cer,
    "wer": avg_wer,
    "bow_hits": avg_bow_hits,
    "bow_extras": avg_bow_extras
}

write_json_file(metrics_aggr, OUTPUT_DIR / f"metrics_aggr_step_{eval_cp.step_idx}.json")

# Detailed results
metrics_lists = {
    "cer": [str(val) for val in cer_list],
    "wer": [str(val) for val in wer_list],
    "bow_hits": [str(val) for val in bow_hits_list],
    "bow_extras": [str(val) for val in bow_extras_list]
}

write_json_file(metrics_lists, OUTPUT_DIR / f"metrics_lists_step_{eval_cp.step_idx}.json")

# Write ground text for reference
write_list_to_text_file(gt_list, OUTPUT_DIR / "ground_truth.txt")

# Write prediction for reference
pred_list = [pred[task] for pred in pred_list]
write_list_to_text_file(pred_list, OUTPUT_DIR / f"prediction_step_{eval_cp.step_idx}.txt")

