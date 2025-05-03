import sys
from pathlib import Path

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO
from htrflow.utils.geometry import Bbox
from shapely.geometry import Polygon
from PIL.Image import Image as PILImage

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from src.data_processing.visual_tasks import coords_to_bbox_xyxy
from pipelines.steps.reading_order import sort_top_down_left_right


def object_detection(od_model: YOLO, image: PILImage, device: str = "cpu") -> list[Bbox]:
    result_od = od_model.predict(image, verbose=False, device=device)
    bboxes_raw = result_od[0].boxes.xyxy
    bboxes = [Bbox(*bbox) for bbox in bboxes_raw]

    if len(bboxes) == 0:
        return []

    # Sort regions
    sorted_region_indices = sort_top_down_left_right(bboxes)
    sorted_bboxes = [bboxes[i] for i in sorted_region_indices]
    return sorted_bboxes


def line_seg(line_seg_model: YOLO, region_img: PILImage, device: str = "cpu") -> list[PILImage]:
    results_line_seg = line_seg_model(region_img, verbose=False, device=device)
    
    if results_line_seg[0].masks is None:
        return []

    # Sort masks
    masks               = [Polygon([(int(point[0]), int(point[1])) for point in mask]) for mask in results_line_seg[0].masks.xy]
    line_bboxes         = [Bbox(*coords_to_bbox_xyxy(line)) for line in masks if len(line) > 0]
    sorted_line_indices = sort_top_down_left_right(line_bboxes)
    sorted_line_masks   = [masks[i] for i in sorted_line_indices]

    return sorted_line_masks


def line_od(line_od_model: YOLO, image: PILImage, device: str = "cpu") -> list[Bbox]:
    results_line_od = line_od_model.predict(image, verbose=False, device=device)
    line_bboxes_raw = results_line_od[0].boxes.xyxy
    line_bboxes = [Bbox(*bbox) for bbox in line_bboxes_raw]

    if len(line_bboxes) == 0:
        return []

    # Sort lines
    sorted_line_indices = sort_top_down_left_right(line_bboxes)
    sorted_line_bboxes = [line_bboxes[i] for i in sorted_line_indices]
    return sorted_line_bboxes


def ocr(ocr_model: VisionEncoderDecoderModel, processor: TrOCRProcessor, line_images: list[PILImage], device: str = "cpu") -> list[str]:
    pixel_values    = processor(images=line_images, return_tensors="pt").pixel_values.to(device)
    generated_ids   = ocr_model.generate(inputs=pixel_values)
    line_trans      = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return line_trans
