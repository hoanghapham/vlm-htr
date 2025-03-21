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

from src.train import load_best_checkpoint, load_last_checkpoint, Checkpoint
from src.data_process.utils import create_dset_from_paths
from src.data_process.trocr import create_trocr_collate_fn
from src.data_process.florence import RunningTextDataset

from src.logger import CustomLogger
from src.file_tools import write_json_file, write_list_to_text_file, read_json_file
#%%

parser = ArgumentParser()
parser.add_argument("--model-name", required=True)
parser.add_argument("--input-dir", required=True)
parser.add_argument("--batch-size", default=15)
parser.add_argument("--use-split-info", default="false")
parser.add_argument("--load-checkpoint", default="best", choices=["last", "best", "vanilla"])
args = parser.parse_args()

# args = parser.parse_args([
#     "--model-name", "trocr_base__ft_vanilla",
#     "--input-dir", str(PROJECT_DIR / "data/hovratt_line"),
#     "--use-split-info", "false",
#     "--load-checkpoint", "vanilla",
#     "--batch-size", "2"
# ])


MODEL_NAME      = args.model_name
INPUT_DIR       = Path(args.input_dir)
USE_SPLIT_INFO  = args.use_split_info == "true"
LOAD_CHECKPOINT = args.load_checkpoint
BATCH_SIZE      = int(args.batch_size)
    
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REMOTE_MODEL_PATH = "microsoft/trocr-base-handwritten"
LOCAL_MODEL_PATH = PROJECT_DIR / "models" / MODEL_NAME

OUTPUT_DIR = PROJECT_DIR / "output" / MODEL_NAME / INPUT_DIR.stem
if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True)

logger = CustomLogger(f"eval__{MODEL_NAME}__{INPUT_DIR.stem}", log_to_local=True)


#%%
# Load model
processor = TrOCRProcessor.from_pretrained(REMOTE_MODEL_PATH)
model = VisionEncoderDecoderModel.from_pretrained(REMOTE_MODEL_PATH).to(DEVICE)

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

collate_fn = create_trocr_collate_fn(processor, DEVICE)
test_loader = DataLoader(test_data, collate_fn=collate_fn, batch_size=BATCH_SIZE)

logger.info(f"Total samples: {len(test_data):,}, batch size: {BATCH_SIZE}, total batches: {len(test_loader):,}")

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

for inputs in tqdm(test_loader, desc="Evaluate"):

    groundtruths = [data["answer"] for data in test_data.select(range(range_start, range_end))]

    generated_ids = model.generate(inputs=inputs["pixel_values"])
    preds = processor.batch_decode(generated_ids, skip_special_tokens=True)

    for pred, gt in zip(preds, groundtruths):
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
    range_end = min(len(test_data), range_end + BATCH_SIZE)


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
write_list_to_text_file(pred_list, OUTPUT_DIR / "prediction.txt")

