#%%
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from argparse import ArgumentParser

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from tqdm import tqdm
from htrflow.evaluate import CER, WER, BagOfWords

from src.train import load_best_checkpoint, load_last_checkpoint, Checkpoint
from src.data_process.utils import create_dset_from_paths
from src.data_process.florence import RunningTextDataset

from src.logger import CustomLogger
from src.file_tools import write_json_file, write_list_to_text_file, read_json_file
#%%

parser = ArgumentParser()
parser.add_argument("--model-name", required=True)
parser.add_argument("--input-dir", required=True)
parser.add_argument("--use-split-info", default="false")
parser.add_argument("--load-checkpoint", default="best", choices=["last", "best", "vanilla"])
parser.add_argument("--user-prompt", required=True, default="<SwedishHTR>Print out the text in this image")
args = parser.parse_args()

# args = parser.parse_args([
#     "--model-name", "florence_base__ft_htr_line",
#     "--input-dir", str(PROJECT_DIR / "data/hovratt_line"),
#     "--use-split-info", "true"
# ])

# Setup paths
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = args.model_name
LOCAL_MODEL_PATH = PROJECT_DIR / "models" / MODEL_NAME
REMOTE_MODEL_PATH = "microsoft/Florence-2-base-ft"

INPUT_DIR = Path(args.input_dir)
LOAD_CHECKPOINT = args.load_checkpoint
USE_SPLIT_INFO = args.use_split_info == "true"
USER_PROMPT = args.user_prompt
OUTPUT_DIR = PROJECT_DIR / "output" / MODEL_NAME / INPUT_DIR.stem

if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True)

# Logger
logger = CustomLogger(f"eval__{MODEL_NAME}__{INPUT_DIR.stem}", log_to_local=True)

#%%
# Load model
logger.info("Load model")
processor = AutoProcessor.from_pretrained(REMOTE_MODEL_PATH, trust_remote_code=True, device_map=DEVICE)
model = AutoModelForCausalLM.from_pretrained(REMOTE_MODEL_PATH, trust_remote_code=True, device_map=DEVICE)

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

if USE_SPLIT_INFO:
    split_info_fp = INPUT_DIR / "split_info.json"
    split_info = read_json_file(split_info_fp)
    test_page_names = [Path(path).stem for path in split_info["test"]]
    test_data_paths = [path for path in INPUT_DIR.glob("*") if path.is_dir() and path.name in test_page_names]
else:
    test_data_paths = [path for path in INPUT_DIR.glob("*") if path.is_dir()]

test_data = create_dset_from_paths(test_data_paths, RunningTextDataset)

logger.info(f"Total test samples: {len(test_data)}")

# Evaluate
#%%
cer = CER()
wer = WER()
bow = BagOfWords()

logger.info(f"User prompt: {USER_PROMPT}")

cer_list = []
wer_list = []
bow_hits_list = []
bow_extras_list = []
gt_list = []
pred_list = []


# Can only process one image at a time due to post_process_generation task requiring image size,
# but image size varies
for inputs in tqdm(test_data, total=len(test_data), desc="Evaluate"):

    image = inputs["image"]
    troundtruth = inputs["answer"]
    inputs = processor(text=USER_PROMPT, images=image, return_tensors="pt").to(DEVICE)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
        num_beams=3,
    )

    output = processor.batch_decode(generated_ids, skip_special_tokens=False)[0] # 0 because current "batch" size is 1
    pred = processor.post_process_generation(output, task="<SwedishHTR>", image_size=image.size)

    # Calcualte metrics
    cer_value = cer.compute(pred["<SwedishHTR>"], troundtruth)["cer"]
    wer_value = wer.compute(pred["<SwedishHTR>"], troundtruth)["wer"]
    bow_hits_value = bow.compute(pred["<SwedishHTR>"], troundtruth)["bow_hits"]
    bow_extras_value = bow.compute(pred["<SwedishHTR>"], troundtruth)["bow_extras"]

    # Append results
    cer_list.append(cer_value)
    wer_list.append(wer_value)
    bow_hits_list.append(bow_hits_value)
    bow_extras_list.append(bow_extras_value)
    gt_list.append(troundtruth)
    pred_list.append(pred)

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
    "epoch": eval_cp.epoch,
    "train_loss": eval_cp.train_loss,
    "val_loss": eval_cp.val_loss,
    "cer": avg_cer,
    "wer": avg_wer,
    "bow_hits": avg_bow_hits,
    "bow_extras": avg_bow_extras
}

write_json_file(metrics_aggr, OUTPUT_DIR / "metrics_aggr.json")

# Detailed results
metrics_lists = {
    "cer": [str(val) for val in cer_list],
    "wer": [str(val) for val in wer_list],
    "bow_hits": [str(val) for val in bow_hits_list],
    "bow_extras": [str(val) for val in bow_extras_list]
}

write_json_file(metrics_lists, OUTPUT_DIR / "metrics_lists.json")

# Write ground text for reference
write_list_to_text_file(gt_list, OUTPUT_DIR / "ground_truth.txt")

# Write prediction for reference
pred_list = [line["<SwedishHTR>"] for line in pred_list]
write_list_to_text_file(pred_list, OUTPUT_DIR / "prediction.txt")

