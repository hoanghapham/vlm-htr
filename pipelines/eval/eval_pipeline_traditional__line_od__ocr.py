#%%
import sys
from pathlib import Path
from argparse import ArgumentParser

import torch
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from htrflow.utils.layout import estimate_printspace
from htrflow.utils.geometry import Bbox
from htrflow.evaluate import CER, WER, BagOfWords

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from src.file_tools import list_files, write_json_file, write_text_file, read_json_file
from src.data_processing.visual_tasks import IMAGE_EXTENSIONS, crop_image, bbox_xyxy_to_coords
from src.data_processing.utils import XMLParser
from src.post_process import topdown_left_right
from src.logger import CustomLogger
from src.evaluation.utils import Ratio
from src.evaluation.ocr_metrics import compute_ocr_metrics


# Setup
parser = ArgumentParser()
parser.add_argument("--split-type", required=True, default="mixed", choices=["mixed", "sbs"])
parser.add_argument("--batch-size", default=6)
parser.add_argument("--debug", required=False, default="false")
args = parser.parse_args()

# args = parser.parse_args([
#     "--split-type", "mixed",
#     "--batch-size", "2",
#     "--debug", "true",
# ])


SPLIT_TYPE      = args.split_type
BATCH_SIZE      = int(args.batch_size)
TEST_DATA_DIR   = PROJECT_DIR / f"data/page/{SPLIT_TYPE}/test/"
OUTPUT_DIR      = PROJECT_DIR / f"evaluations/pipeline_traditional__{SPLIT_TYPE}__line_od__ocr"
DEBUG           = args.debug == "true"

img_paths = list_files(TEST_DATA_DIR, IMAGE_EXTENSIONS)
xml_paths = list_files(TEST_DATA_DIR, [".xml"])

if DEBUG:
    img_paths = [img_paths[0], img_paths[704]]
    xml_paths = [xml_paths[0], xml_paths[704]]


#%%

logger = CustomLogger(f"pl_trad__{SPLIT_TYPE}__2steps")

# Load models
model_line_od  = YOLO(PROJECT_DIR / f"models/trained/yolo11m__{SPLIT_TYPE}__page__line_od/weights/best.pt")

DEVICE              = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REMOTE_MODEL_PATH   = "microsoft/trocr-base-handwritten"
LOCAL_MODEL_PATH    = PROJECT_DIR / f"models/trained/trocr_base__{SPLIT_TYPE}__line_seg__ocr/best"
processor           = TrOCRProcessor.from_pretrained(REMOTE_MODEL_PATH)
model_ocr           = VisionEncoderDecoderModel.from_pretrained(LOCAL_MODEL_PATH).to(DEVICE)

#%%
xml_parser = XMLParser()
cer = CER()
wer = WER()
bow = BagOfWords()

cer_list = []
wer_list = []
bow_hits_list = []
bow_extras_list = []


for img_idx, (img_path, xml_path) in enumerate(zip(img_paths, xml_paths)):
    # logger.info(f"Processing image {img_idx+1}/{len(img_paths)}")

    # Skip if the file is already processed
    img_metric_path = OUTPUT_DIR / (Path(img_path).stem + "__metrics.json")
    if img_metric_path.exists():
        logger.info(f"Skip: {img_path.name}")
        img_metric = read_json_file(img_metric_path)
        cer_list.append(Ratio(*img_metric["cer"]["str"].split("/")))
        wer_list.append(Ratio(*img_metric["wer"]["str"].split("/")))
        bow_hits_list.append(Ratio(*img_metric["bow_hits"]["str"].split("/")))
        bow_extras_list.append(Ratio(*img_metric["bow_extras"]["str"].split("/")))
        continue

    logger.info(f"Image {img_idx}/{len(img_paths)}: {img_path.name}")

    image = Image.open(img_path).convert("RGB")
    gt_lines = xml_parser.get_lines(xml_path)
    printspace = estimate_printspace(np.array(image))

    ## line OD
    logger.info("Line detection")
    results_line_od = model_line_od.predict(image, verbose=False, device=DEVICE)
    line_bboxes_raw = results_line_od[0].boxes.xyxy
    line_bboxes = [Bbox(*bbox) for bbox in line_bboxes_raw]

    if len(line_bboxes) == 0:
        logger.warning("No line detected")
        cer_list.append(Ratio(0, 0))
        wer_list.append(Ratio(0, 0))
        bow_hits_list.append(Ratio(0, 0))
        bow_extras_list.append(Ratio(0, 0))
        continue

    # Sort lines
    sorted_line_indices = topdown_left_right(line_bboxes)
    sorted_line_bboxes = [line_bboxes[i] for i in sorted_line_indices]


    ## OCR
    logger.info("Text recognition")

    iterator = list(range(0, len(sorted_line_bboxes), BATCH_SIZE))
    page_trans = []

    for i in tqdm(iterator, total=len(iterator), unit="batch"):

        # Create a batch of cropped line images
        batch = sorted_line_bboxes[i:i+BATCH_SIZE]
        cropped_line_imgs = []

        # Cut line segs from line images
        for line_bbox in batch:
            crop_coords = bbox_xyxy_to_coords(line_bbox)
            cropped_line_img = crop_image(image, crop_coords)

            cropped_line_imgs.append(cropped_line_img)

        # Batch inference
        pixel_values    = processor(images=cropped_line_imgs, return_tensors="pt").pixel_values.to(DEVICE)
        generated_ids   = model_ocr.generate(inputs=pixel_values)
        line_trans      = processor.batch_decode(generated_ids, skip_special_tokens=True)
        page_trans += line_trans

    # Stitching. Transcriptions are already in the right order
    pred_text = ""

    for line in page_trans:
        pred_text += line + " "

    # Output in .hyp extension to be used with E2EHTREval
    write_text_file(pred_text, OUTPUT_DIR / (Path(img_path).stem + ".hyp"))

    # Write ground truth in .ref extension to be used with E2EHTREval
    gt_text = ""

    for line in gt_lines:
        gt_text += line["transcription"] + " "

    write_text_file(gt_text, OUTPUT_DIR / (Path(img_path).stem + ".ref"))
    
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

#%%
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
