#%%
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from tqdm import tqdm
from htrflow.evaluate import CER, WER, BagOfWords

from src.train import load_checkpoint
from src.data_processing.trocr import create_collate_fn
from src.data_processing.utils import load_arrow_datasets
from src.evaluation.utils import Ratio
from src.file_tools import write_json_file, write_list_to_text_file
from src.logger import CustomLogger
#%%

parser = ArgumentParser()
parser.add_argument("--model-name", required=True)
parser.add_argument("--data-dir", required=True)
parser.add_argument("--batch-size", default=15)
parser.add_argument("--checkpoint", default="best", choices=["last", "best", "vanilla", "specific"])
parser.add_argument("--checkpoint-path", required=False)
parser.add_argument("--debug", default="false")
args = parser.parse_args()

# args = parser.parse_args([
#     "--model-name", "trocr_base__mixed__line_seg__ocr",
#     "--data-dir", str(PROJECT_DIR / "data/line_seg/mixed/test"),
#     "--checkpoint", "best",
#     "--batch-size", "2",
#     "--debug", "true"
# ])


MODEL_NAME      = args.model_name
DATA_DIR        = Path(args.data_dir)
BATCH_SIZE      = int(args.batch_size)
CHECKPOINT      = args.checkpoint
CHECKPOINT_PATH = args.checkpoint_path
DEBUG           = args.debug == "true"
MAX_ITERS       = 5

REMOTE_MODEL_PATH   = "microsoft/trocr-base-handwritten"
LOCAL_MODEL_PATH    = PROJECT_DIR / "models/trained" / MODEL_NAME
EVAL_DIR            = PROJECT_DIR / "evaluations" / MODEL_NAME
DEVICE              = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not EVAL_DIR.exists():
    EVAL_DIR.mkdir(parents=True)

logger = CustomLogger(f"eval__{MODEL_NAME}", log_to_local=True)


#%%
# Load model
processor   = TrOCRProcessor.from_pretrained(REMOTE_MODEL_PATH)
model       = VisionEncoderDecoderModel.from_pretrained(REMOTE_MODEL_PATH).to(DEVICE)


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
logger.info("Load test data")
collate_fn      = create_collate_fn(processor, DEVICE)
test_dataset    = load_arrow_datasets(DATA_DIR)
test_loader     = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE)

logger.info(f"Total samples: {len(test_dataset):,}, batch size: {BATCH_SIZE}, total batches: {len(test_loader):,}")

#%%
cer = CER()
wer = WER()
bow = BagOfWords()

cer_list = []
wer_list = []
bow_hits_list = []
bow_extras_list = []
gt_list = []
pred_list = []

range_start = 0
range_end = BATCH_SIZE
counter = 0


for inputs in tqdm(test_loader, desc="Evaluate"):

    groundtruths = [data["transcription"] for data in test_dataset.select(range(range_start, range_end))]

    generated_ids = model.generate(inputs=inputs["pixel_values"])
    preds = processor.batch_decode(generated_ids, skip_special_tokens=True)

    for pred, gt in zip(preds, groundtruths):

        if pred == gt == "":
            cer_value           = Ratio(0, 0)
            wer_value           = Ratio(0, 0)
            bow_hits_value      = Ratio(0, 0)
            bow_extras_value    = Ratio(0, 0)
        elif pred != gt and (pred == "" or gt == ""):
            value = max(len(pred), len(gt))
            cer_value           = Ratio(value, value)
            wer_value           = Ratio(value, value)
            bow_hits_value      = Ratio(value, value)
            bow_extras_value    = Ratio(value, value)
        else:
            cer_value           = cer.compute(pred, gt)["cer"]
            wer_value           = wer.compute(pred, gt)["wer"]
            bow_hits_value      = bow.compute(pred, gt)["bow_hits"]
            bow_extras_value    = bow.compute(pred, gt)["bow_extras"]

        # Append results
        cer_list.append(cer_value)
        wer_list.append(wer_value)
        bow_hits_list.append(bow_hits_value)
        bow_extras_list.append(bow_extras_value)
        gt_list.append(gt)
        pred_list.append(pred)
    
    range_start += BATCH_SIZE
    range_end = min(len(test_dataset), range_end + BATCH_SIZE)

    if DEBUG:
        counter += 1
        if counter >= MAX_ITERS:
            break


# %%
avg_cer = float(sum(cer_list))
avg_wer = float(sum(wer_list))
avg_bow_hits = float(sum(bow_hits_list))
avg_bow_extras = float(sum(bow_extras_list))

logger.info(f"Avg. CER: {avg_cer:.4f}, Avg. WER: {avg_wer:.4f}")
logger.info(f"Avg. BoW hits: {avg_bow_hits:.4f}, Avg. BoW extras: {avg_bow_extras:.4f}")

#%%

# Save results
# Avg metrics

logger.info(f"Save result to {EVAL_DIR}")

metrics_aggr = {
    "cer": avg_cer,
    "wer": avg_wer,
    "bow_hits": avg_bow_hits,
    "bow_extras": avg_bow_extras
}

if CHECKPOINT == "vanilla":
    suffix = "vanilla"
else:
    suffix = "step_" + str(cp_train_metrics["step_idx"]).zfill(10)


write_json_file(metrics_aggr, EVAL_DIR / f"metrics_aggr_{suffix}.json")

# Detailed results
metrics_lists = {
    "cer": [str(val) for val in cer_list],
    "wer": [str(val) for val in wer_list],
    "bow_hits": [str(val) for val in bow_hits_list],
    "bow_extras": [str(val) for val in bow_extras_list]
}

write_json_file(metrics_lists, EVAL_DIR / f"metrics_lists_{suffix}.json")

# Write ground text for reference
write_list_to_text_file(gt_list, EVAL_DIR / "ground_truth.txt")

# Write prediction for reference
write_list_to_text_file(pred_list, EVAL_DIR / f"prediction_{suffix}.txt")

