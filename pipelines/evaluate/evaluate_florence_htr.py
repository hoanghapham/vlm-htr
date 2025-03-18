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

from src.logger import CustomLogger
from src.train import load_best_checkpoint
from src.file_tools import write_json_file, write_list_to_text_file, read_json_file, write_ndjson_file
from src.tasks.utils import create_dset_from_paths




#%%

parser = ArgumentParser()
parser.add_argument("--model-name", required=True)
parser.add_argument("--input-dir", required=True)
# parser.add_argument("--output-dir", required=True)
args = parser.parse_args([])

# args = parser.parse_args([
#     "--model-name", "florence-2-base-ft-htr-line",
#     "--input-dir", str(PROJECT_DIR/"data/poliskammare_line")
# ])

# Setup paths
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = args.model_name
LOCAL_MODEL_PATH = PROJECT_DIR / "models" / MODEL_NAME
REMOTE_MODEL_PATH = "microsoft/Florence-2-base-ft"

INPUT_DIR = Path(args.input_dir)
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

# Load best checkpoint

best_state = load_best_checkpoint(LOCAL_MODEL_PATH, DEVICE)
model.load_state_dict(best_state["model_state_dict"])
best_epoch = best_state["epoch"]
best_loss = best_state["loss"]
logger.info(f"Load best checkpoint: epoch {best_epoch}, loss: {best_loss:.4f}")

# Set model to evaluation mode
model.eval()

#%%
logger = CustomLogger(f"eval__{MODEL_NAME}__{INPUT_DIR.stem}", log_to_local=True)
# Load split info
logger.info("Load test data")
split_info_fp = INPUT_DIR / "split_info.json"

# Load data to test
if split_info_fp.exists():
    split_info = read_json_file(split_info_fp)
    test_page_names = [Path(path).stem for path in split_info["test"]]
    test_data_paths = [path for path in INPUT_DIR.glob("*") if path.is_dir() and path.name in test_page_names]
else:
    test_data_paths = [path for path in INPUT_DIR.glob("*") if path.is_dir()]

test_data = create_dset_from_paths(test_data_paths)

logger.info(f"Test samples: {len(test_data)}")

# Evaluate
#%%
cer = CER()
wer = WER()
bow = BagOfWords()
prompt = "<SwedishHTR>Print out the text in this image"

logger.info(f"Test user prompt: {prompt}")

cer_list = []
wer_list = []
bow_hits_list = []
bow_extras_list = []
transcr_gt_list = []
transcr_pred_list = []


# Can only process one image at a time due to post_process_generation task requiring image size,
# but image size varies
for inputs in tqdm(test_data, unit="line", total=len(test_data), desc="Evaluate"):

    image = inputs["image"]
    transcription_gt = inputs["answer"]
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
        num_beams=3,
    )

    output_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0] # 0 because current "batch" size is 1
    pred = processor.post_process_generation(output_text, task="<SwedishHTR>", image_size=image.size)
    # transcr_pred.append(pred)

    cer_value = cer.compute(pred["<SwedishHTR>"], transcription_gt)["cer"]
    wer_value = wer.compute(pred["<SwedishHTR>"], transcription_gt)["wer"]
    bow_hits_value = bow.compute(pred["<SwedishHTR>"], transcription_gt)["bow_hits"]
    bow_extras_value = bow.compute(pred["<SwedishHTR>"], transcription_gt)["bow_extras"]

    cer_list.append(cer_value)
    wer_list.append(wer_value)
    bow_hits_list.append(bow_hits_value)
    bow_extras_list.append(bow_extras_value)
    transcr_gt_list.append(transcription_gt)
    transcr_pred_list.append(pred)

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
    "best_epoch": best_epoch,
    "best_loss": best_loss,
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

# Predicted text
write_ndjson_file(transcr_gt_list, OUTPUT_DIR / "ground_truth.json")

pred_list = [line["<SwedishHTR>"] for line in transcr_pred_list]
write_ndjson_file(pred_list, OUTPUT_DIR / "prediction.txt")


# %%
