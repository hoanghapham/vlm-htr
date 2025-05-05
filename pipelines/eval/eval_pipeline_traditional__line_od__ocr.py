#%%
import sys
from pathlib import Path
from argparse import ArgumentParser

import torch
from ultralytics import YOLO
from tqdm import tqdm
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from src.file_tools import list_files, write_json_file, write_text_file, read_json_file
from src.data_processing.visual_tasks import IMAGE_EXTENSIONS, crop_image, bbox_xyxy_to_coords
from src.data_processing.utils import XMLParser
from src.logger import CustomLogger
from src.evaluation.utils import Ratio
from src.evaluation.ocr_metrics import compute_ocr_metrics
from pipelines.steps.traditional import line_od, ocr
from pipelines.steps.general import read_img_metrics

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
line_od_model  = YOLO(PROJECT_DIR / f"models/trained/yolo11m__{SPLIT_TYPE}__page__line_od/weights/best.pt")

DEVICE              = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REMOTE_MODEL_PATH   = "microsoft/trocr-base-handwritten"
LOCAL_MODEL_PATH    = PROJECT_DIR / f"models/trained/trocr_base__{SPLIT_TYPE}__line_bbox__ocr/best"
processor           = TrOCRProcessor.from_pretrained(REMOTE_MODEL_PATH)
ocr_model           = VisionEncoderDecoderModel.from_pretrained(LOCAL_MODEL_PATH).to(DEVICE)

#%%
xml_parser = XMLParser()
cer_list = []
wer_list = []
bow_hits_list = []
bow_extras_list = []



#%%
for img_idx, (img_path, xml_path) in enumerate(zip(img_paths, xml_paths)):
    # logger.info(f"Processing image {img_idx+1}/{len(img_paths)}")

    # Skip if the file is already processed
    img_metric_path = OUTPUT_DIR / (Path(img_path).stem + "__metrics.json")
    if img_metric_path.exists() and not DEBUG:
        logger.info(f"Skip: {img_path.name}")
        cerlist, werlist, bow_hits_list, bow_extras_list = read_img_metrics(
            img_metric_path, cer_list, wer_list, bow_hits_list, bow_extras_list)
        continue

    logger.info(f"Image {img_idx}/{len(img_paths)}: {img_path.name}")

    image = Image.open(img_path).convert("RGB")
    gt_lines = xml_parser.get_lines(xml_path)

    ## line OD
    logger.info("Line detection")
    sorted_line_bboxes = line_od(line_od_model, image, DEVICE)

    if len(sorted_line_bboxes) == 0:
        logger.warning("No line detected")
        continue

    ## OCR
    logger.info("Text recognition")

    iterator = list(range(0, len(sorted_line_bboxes), BATCH_SIZE))
    page_trans = []

    for i in tqdm(iterator, total=len(iterator), unit="batch"):

        # Create a batch of cropped line images
        batch = sorted_line_bboxes[i:i+BATCH_SIZE]
        batch_line_imgs = []

        # Cut line segs from line images
        for line_bbox in batch:
            crop_coords = bbox_xyxy_to_coords(line_bbox)
            line_imgs = crop_image(image, crop_coords)
            batch_line_imgs.append(line_imgs)

        # Batch inference
        line_trans = ocr(ocr_model, processor, batch_line_imgs, device=DEVICE)
        page_trans += line_trans

    # Stitching. Transcriptions are already in the right order
    # Output in .hyp extension to be used with E2EHTREval
    pred_text = " ".join(page_trans)
    write_text_file(pred_text, OUTPUT_DIR / (Path(img_path).stem + ".hyp"))

    # Write ground truth in .ref extension to be used with E2EHTREval
    gt_text = " ".join([line["transcription"] for line in gt_lines])
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
