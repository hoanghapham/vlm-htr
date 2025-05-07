#%%
import sys
from pathlib import Path
from argparse import ArgumentParser

import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from src.file_tools import list_files, write_json_file, write_text_file
from src.data_processing.visual_tasks import IMAGE_EXTENSIONS, crop_image
from src.data_processing.utils import XMLParser
from src.evaluation.ocr_metrics import compute_ocr_metrics
from src.logger import CustomLogger
from pipelines.steps.florence import line_od, line_seg, ocr
from pipelines.steps.generic import read_img_metrics


# Setup
parser = ArgumentParser()
parser.add_argument("--split-type", required=True, default="mixed", choices=["mixed", "sbs"])
parser.add_argument("--batch-size", default=6)
parser.add_argument("--device", default="cuda", choices="cpu")
parser.add_argument("--debug", required=False, default="false")
# args = parser.parse_args()

args = parser.parse_args([
    "--split-type", "mixed",
    "--batch-size", "2",
    "--device", "cpu",
    "--debug", "true",
])

SPLIT_TYPE      = args.split_type
BATCH_SIZE      = int(args.batch_size)
DEBUG           = args.debug == "true"
TEST_DATA_DIR   = PROJECT_DIR / f"data/page/{SPLIT_TYPE}/test/"
OUTPUT_DIR      = PROJECT_DIR / f"evaluations/pipeline_florence__{SPLIT_TYPE}__line_od__line_seg__ocr"

img_paths = list_files(TEST_DATA_DIR, IMAGE_EXTENSIONS)
xml_paths = list_files(TEST_DATA_DIR, [".xml"])

if DEBUG:
    img_paths = [img_paths[0], img_paths[704]]
    xml_paths = [xml_paths[0], xml_paths[704]]

#%%

logger = CustomLogger(f"pl_flor__{SPLIT_TYPE}__3steps")

# Load models
if args.device == "cuda":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = args.device
REMOTE_MODEL_PATH   = "microsoft/Florence-2-base-ft"

line_od_model   = AutoModelForCausalLM.from_pretrained(
                    PROJECT_DIR / f"models/trained/florence_base__{SPLIT_TYPE}__page__line_od/best",
                    trust_remote_code=True).to(DEVICE)
line_seg_model   = AutoModelForCausalLM.from_pretrained(
                    PROJECT_DIR / f"models/trained/florence_base__{SPLIT_TYPE}__line_cropped__line_seg/best",
                    trust_remote_code=True).to(DEVICE)
ocr_model       = AutoModelForCausalLM.from_pretrained(
                    PROJECT_DIR / f"models/trained/florence_base__{SPLIT_TYPE}__line_seg__ocr/best",
                    trust_remote_code=True).to(DEVICE)

processor       = AutoProcessor.from_pretrained(REMOTE_MODEL_PATH, trust_remote_code=True, device_map=DEVICE)

#%%
xml_parser = XMLParser()
cer_list = []
wer_list = []
bow_hits_list = []
bow_extras_list = []


# Iterate through images
for img_idx, (img_path, xml_path) in enumerate(zip(img_paths, xml_paths)):

    # Skip if the file is already processed
    img_metric_path = OUTPUT_DIR / (Path(img_path).stem + "__metrics.json")
    if img_metric_path.exists() and not DEBUG:
        logger.info(f"Skip: {img_path.name}")
        cerlist, werlist, bow_hits_list, bow_extras_list = read_img_metrics(
            img_metric_path, cer_list, wer_list, bow_hits_list, bow_extras_list)
        continue

    logger.info(f"Image {img_idx}/{len(img_paths)}: {img_path.name}")
    image       = Image.open(img_path).convert("RGB")


    ## Line OD
    logger.info("Line detection")
    line_od_output = line_od(line_od_model, processor, image, DEVICE)

    ## Line seg then OCR
    logger.info("Line segmentation -> Text recognition")
    page_trans = []

    # Iterate through detected lines
    iterator = list(range(0, len(line_od_output.polygons), BATCH_SIZE))

    for i in tqdm(iterator, total=len(iterator), unit="batch"):

        # Create a batch of cropped line bboxes
        batch = line_od_output.polygons[i:i+BATCH_SIZE]
        batch_line_bbox_imgs = []

        for bbox_polygon in batch:
            batch_line_bbox_imgs.append(crop_image(image, bbox_polygon))

        # Line segmentation
        line_seg_output     = line_seg(line_seg_model, processor, batch_line_bbox_imgs, DEVICE)
        batch_line_seg_imgs = []

        for line_img, mask in zip(batch_line_bbox_imgs, line_seg_output.polygons):
            batch_line_seg_imgs.append(crop_image(line_img, mask))
        
        # OCR
        line_trans = ocr(ocr_model, processor, batch_line_seg_imgs, DEVICE)
        page_trans += line_trans


    # Stitching. Transcriptions are already in the right order
    # Output in .hyp extension to be used with E2EHTREval
    pred_text = " ".join(page_trans)
    write_text_file(pred_text, OUTPUT_DIR / (Path(img_path).stem + ".hyp"))

    # Get lines from xml
    # Write ground truth in .ref extension to be used with E2EHTREval
    gt_lines    = xml_parser.get_lines(xml_path)
    gt_text     = " ".join([line["transcription"] for line in gt_lines])
    write_text_file(gt_text, OUTPUT_DIR / (Path(img_path).stem + ".ref"))

    # Evaluation
    try:
        metrics_ratio   = compute_ocr_metrics(gt_text, pred_text, return_type="ratio")
        metrics_str     = compute_ocr_metrics(gt_text, pred_text, return_type="str")
    except Exception as e:
        logger.exception(e)
        continue

    cer_list.append(metrics_ratio["cer"])
    wer_list.append(metrics_ratio["wer"])
    bow_hits_list.append(metrics_ratio["bow_hits"])
    bow_extras_list.append(metrics_ratio["bow_extras"])

    logger.info(f"CER: {float(metrics_ratio['cer']):.4f}, WER: {float(metrics_ratio['wer']):.4f}, BoW hits: {float(metrics_ratio['bow_hits']):.4f}, BoW extras: {float(metrics_ratio['bow_extras']):.4f}")

    write_json_file(metrics_str, OUTPUT_DIR / (Path(img_path).stem + "__metrics.json"))


# Averaging metrics across all pages
avg_cer = sum(cer_list)
avg_wer = sum(wer_list)
avg_bow_hits = sum(bow_hits_list)
avg_bow_extras = sum(bow_extras_list)

logger.info(f"Avg. CER: {float(avg_cer):.4f}, Avg. WER: {float(avg_wer):.4f}, Avg. BoW hits: {float(avg_bow_hits):.4f}, Avg. BoW extras: {float(avg_bow_extras):.4f}")

avg_metrics = {
    "cer": {"str": str(avg_cer), "float": float(avg_cer)},
    "wer": {"str": str(avg_wer), "float": float(avg_wer)},
    "bow_hits": {"str": str(avg_bow_hits), "float": float(avg_bow_hits)},
    "bow_extras": {"str": str(avg_bow_extras), "float": float(avg_bow_extras)}
}

write_json_file(avg_metrics, OUTPUT_DIR / "avg_metrics.json")
# %%

# from src.visualization import draw_segment_masks

# draw_segment_masks(line_img, [mask])