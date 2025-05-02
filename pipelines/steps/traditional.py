import sys
import os
from pathlib import Path

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from htrflow.utils.geometry import Bbox, Polygon
from PIL.Image import Image as PILImage
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from pipelines.steps.reading_order import topdown_left_right


def line_od(line_od_model: YOLO, image: PILImage, device: str = "cpu") -> list[Bbox]:
    results_line_od = line_od_model.predict(image, verbose=False, device=device)
    line_bboxes_raw = results_line_od[0].boxes.xyxy
    line_bboxes = [Bbox(*bbox) for bbox in line_bboxes_raw]

    if len(line_bboxes) == 0:
        return []

    # Sort lines
    sorted_line_indices = topdown_left_right(line_bboxes)
    sorted_line_bboxes = [line_bboxes[i] for i in sorted_line_indices]
    return sorted_line_bboxes


def ocr(ocr_model: VisionEncoderDecoderModel, processor: TrOCRProcessor, line_images: list[PILImage], device: str = "cpu") -> list[str]:
    pixel_values    = processor(images=line_images, return_tensors="pt").pixel_values.to(device)
    generated_ids   = ocr_model.generate(inputs=pixel_values)
    line_trans      = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return line_trans
