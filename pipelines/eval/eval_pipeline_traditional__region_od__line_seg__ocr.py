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
from src.data_processing.visual_tasks import IMAGE_EXTENSIONS, crop_image, bbox_xyxy_to_coords, coords_to_bbox_xyxy,
from src.data_processing.utils import XMLParser
from src.post_process import order_bboxes
from src.logger import CustomLogger
from src.evaluation.utils import Ratio


# Setup
parser = ArgumentParser()
parser.add_argument("--split-type", required=True, default="mixed", choices=["mixed", "sbs"])
parser.add_argument("--batch-size", default=6)
parser.add_argument("--debug", required=False, default="false")
args = parser.parse_args()

SPLIT_TYPE      = args.split_type
BATCH_SIZE      = int(args.batch_size)
TEST_DATA_DIR   = PROJECT_DIR / f"data/page/{SPLIT_TYPE}/test/"
OUTPUT_DIR      = PROJECT_DIR / f"evaluations/pipeline_traditional__{SPLIT_TYPE}__region_od__line_seg__ocr"
DEBUG           = args.debug == "true"

img_paths = list_files(TEST_DATA_DIR, IMAGE_EXTENSIONS)
xml_paths = list_files(TEST_DATA_DIR, [".xml"])

if DEBUG:
    img_paths = [img_paths[0], img_paths[704]]
    xml_paths = [xml_paths[0], xml_paths[704]]

#%%

logger = CustomLogger(f"pl_trad__{SPLIT_TYPE}__3steps")

# Load models
model_region_od = YOLO(PROJECT_DIR / f"models/trained/yolo11m__{SPLIT_TYPE}__page__region_od/weights/best.pt")
model_line_seg  = YOLO(PROJECT_DIR / f"models/trained/yolo11m_seg__{SPLIT_TYPE}__region__line_seg/weights/best.pt")

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


#%%

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

    ## Region OD
    logger.info("Region detection")
    results_region_od = model_region_od.predict(image, verbose=False, device=DEVICE)
    region_bboxes_raw = results_region_od[0].boxes.xyxy
    region_bboxes = [Bbox(*bbox) for bbox in region_bboxes_raw]

    # Sort regions
    sorted_region_indices = order_bboxes(region_bboxes, printspace, True)
    sorted_region_bboxes = [region_bboxes[i] for i in sorted_region_indices]


    ## Line seg
    logger.info("Line segmentation")
    cropped_regions = []
    region_line_masks = []

    for bbox in sorted_region_bboxes:
        # Crop image to region
        crop_coords = bbox_xyxy_to_coords(bbox)
        region_img = crop_image(image, crop_coords)
        cropped_regions.append(region_img)

        # Segment lines
        results_line_seg = model_line_seg(region_img, verbose=False, device=DEVICE)
        if results_line_seg[0].masks is None:
            continue

        masks = results_line_seg[0].masks.xy

        # Sort masks
        line_bboxes = [Bbox(*coords_to_bbox_xyxy(line.astype(int))) for line in masks if len(line) > 0]
        sorted_line_indices = order_bboxes(line_bboxes, printspace, False)
        sorted_line_masks = [masks[i] for i in sorted_line_indices]

        region_line_masks.append(sorted_line_masks)

    # OCR
    logger.info("Text recognition")
    page_trans = []

    for region_idx, (region_img, masks) in enumerate(zip(cropped_regions, region_line_masks)):
        region_trans = []

        iterator = list(range(0, len(masks), BATCH_SIZE))
        
        for i in tqdm(iterator, total=len(iterator), unit="batch", desc=f"Region {region_idx}/{len(cropped_regions)}"):

            # Create a batch of cropped line images
            batch = masks[i:i+BATCH_SIZE]
            cropped_line_imgs = []

            # Cut line segs from region images
            for mask in batch:
                cropped_line_seg = cr(region_img, mask.astype(int))
                cropped_line_imgs.append(cropped_line_seg)

            # Batch inference
            pixel_values    = processor(images=cropped_line_imgs, return_tensors="pt").pixel_values.to(DEVICE)
            generated_ids   = model_ocr.generate(inputs=pixel_values)
            line_trans      = processor.batch_decode(generated_ids, skip_special_tokens=True)
            region_trans += line_trans

        page_trans.append(region_trans)

    # Stitching. Transcriptions are already in the right order
    pred_text = ""

    for region_lines in page_trans:
        for line in region_lines:
            pred_text += line + " "

    # Output in .hyp extension to be used with E2EHTREval
    write_text_file(pred_text, OUTPUT_DIR / (Path(img_path).stem + ".hyp"))

    # Write ground truth in .ref extension to be used with E2EHTREval
    gt_text = ""

    for line in gt_lines:
        gt_text += line["transcription"] + " "

    write_text_file(gt_text, OUTPUT_DIR / (Path(img_path).stem + ".ref"))
    
    # Evaluation
    try:
        cer_value = cer.compute(gt_text, pred_text)["cer"]
        wer_value = wer.compute(gt_text, pred_text)["wer"]
        bow_hits_value = bow.compute(gt_text, pred_text)["bow_hits"]
        bow_extras_value = bow.compute(gt_text, pred_text)["bow_extras"]
    except Exception as e:
        logger.exception(e)
        continue

    cer_list.append(cer_value)
    wer_list.append(wer_value)
    bow_hits_list.append(bow_hits_value)
    bow_extras_list.append(bow_extras_value)

    page_metrics = {
        "cer": {"str": str(cer_value), "float": float(cer_value)},
        "wer": {"str": str(wer_value), "float": float(wer_value)},
        "bow_hits": {"str": str(bow_hits_value), "float": float(bow_hits_value)},
        "bow_extras": {"str": str(bow_extras_value), "float": float(bow_extras_value)}
    }

    logger.info(f"CER: {float(cer_value):.4f}, WER: {float(wer_value):.4f}, BoW hits: {float(bow_hits_value):.4f}, BoW extras: {float(bow_extras_value):.4f}")

    write_json_file(page_metrics, OUTPUT_DIR / (Path(img_path).stem + "__metrics.json"))

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
