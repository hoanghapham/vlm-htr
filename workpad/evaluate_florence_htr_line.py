#%%
import sys
import json
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.append(str(PROJECT_DIR))

import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoProcessor
from datasets import load_dataset, load_from_disk, concatenate_datasets
from tqdm import tqdm
from htrflow.evaluate import CER, WER, BagOfWords

from src.logger import CustomLogger

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOCAL_MODEL_PATH = PROJECT_DIR / "models/florence-2-base-ft-htr-line/"
REMOTE_MODEL_PATH = "microsoft/Florence-2-base-ft"

logger = CustomLogger("eval_florence_2_ft_line")

# Load model
processor = AutoProcessor.from_pretrained(REMOTE_MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(REMOTE_MODEL_PATH, trust_remote_code=True)

#%%
# Load checkpoint
checkpoint_path = PROJECT_DIR / "models/florence-2-base-ft-htr-line/checkpoint_epoch_000.pt"
model_info = torch.load(checkpoint_path, weights_only=True, map_location=torch.device(DEVICE))

model.load_state_dict(model_info["model_state_dict"])
model.eval()

#%%
# Load split info
DATA_DIR = PROJECT_DIR / "data/poliskammare_line_test"

with open(DATA_DIR / "split_info.json", "r") as f:
    split_info = json.load(f)

# copy from remote
# for path in split_info["test"][:10]:
#     if not (DATA_DIR / Path(path).stem).exists():
#         subprocess.run(["scp", "-r", "uppmax:" + path, DATA_DIR])

# Load test data
test_page_names = [Path(path).stem for path in split_info["test"]]
test_data_paths = [path for path in DATA_DIR.glob("*") if path.is_dir() and path.name in test_page_names]

test_data_list = []

for path in test_data_paths:
    tmp = load_from_disk(path)
    test_data_list.append(tmp)

test_data = concatenate_datasets(test_data_list)

logger.info(f"Test samples: {len(test_data)}")

#%%
# Evaluate
cer = CER()
wer = WER()
bow = BagOfWords()

prompt = "<SwedishHTR>Print out the text in this image"

cer_list = []
wer_list = []
bow_hits_list = []
bow_extras_list = []
transcr_gt_list = []
transcr_pred_list = []


for line_data in tqdm(test_data, unit="line", total=len(test_data), desc="Evaluate"):

    image = line_data["image"].convert("RGB")
    transcription_gt = line_data["transcription"]
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
        num_beams=3,
    )

    output_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    transcr_pred = processor.post_process_generation(output_text, task="<SwedishHTR>", image_size=(image.width, image.height))

    cer_value = cer.compute(transcr_pred["<SwedishHTR>"], transcription_gt)["cer"]
    wer_value = wer.compute(transcr_pred["<SwedishHTR>"], transcription_gt)["wer"]
    bow_hits_value = bow.compute(transcr_pred["<SwedishHTR>"], transcription_gt)["bow_hits"]
    bow_extras_value = bow.compute(transcr_pred["<SwedishHTR>"], transcription_gt)["bow_extras"]

    cer_list.append(cer_value)
    wer_list.append(wer_value)
    bow_hits_list.append(bow_hits_value)
    bow_extras_list.append(bow_extras_value)
    transcr_gt_list.append(transcription_gt)
    transcr_pred_list.append(transcr_pred)


#%%

avg_cer = float(sum(cer_list))
avg_wer = float(sum(wer_list))
avg_bow_hits = float(sum(bow_hits_list))
avg_bow_extras = float(sum(bow_extras_list))


logger.info(f"Avg. CER: {avg_cer:.4f}")
logger.info(f"Avg. WER: {avg_wer:.4f}")
logger.info(f"Avg. BoW hits: {avg_bow_hits:.4f}")
logger.info(f"Avg. BoW extrs: {avg_bow_extras:.4f}")

#%%
# Save results
# Avg metrics
metrics_aggr = {
    "cer": avg_cer,
    "wer": avg_wer,
    "bow_hits": avg_bow_hits,
    "bow_extras": avg_bow_extras
}

OUTPUT_DIR = PROJECT_DIR / "output/florence-2-base-ft-htr-line/"

if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True)

with open(OUTPUT_DIR / "metrics_aggr.json", "w") as f:
    json.dump(metrics_aggr, f)


#%%
# Detailed results

metrics_lists = {
    "cer": [str(val) for val in cer_list],
    "wer": [str(val) for val in wer_list],
    "bow_hits": [str(val) for val in bow_hits_list],
    "bow_extras": [str(val) for val in bow_extras_list]
}

with open(OUTPUT_DIR / "metrics_lists.json", "w") as f:
    json.dump(metrics_lists, f)

#%%
# Predicted text
with open(OUTPUT_DIR / "transcription_gt.txt", "w") as f:
    for line in transcr_gt_list:
        f.write(line)
        f.write("\n")

with open(OUTPUT_DIR / "transcr_pred.txt", "w") as f:
    for line in transcr_pred_list:
        f.write(line["<SwedishHTR>"])
        f.write("\n")

# %%
