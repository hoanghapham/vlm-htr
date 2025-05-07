import sys
from pathlib import Path

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO
from htrflow.utils.geometry import Bbox
from shapely.geometry import Polygon
from PIL.Image import Image as PILImage

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from src.data_processing.visual_tasks import polygon_to_bbox_xyxy, bbox_xyxy_to_polygon
from pipelines.steps.postprocess import sort_top_down_left_right, sort_bboxes


class ODOutput():
    def __init__(self, object_bboxes: list[Bbox], object_polygons: list[Polygon]):
        self.bboxes = object_bboxes
        self.polygons = object_polygons


def object_detection(od_model: YOLO, image: PILImage, device: str = "cpu") -> ODOutput:
    result_od = od_model.predict(image, verbose=False, device=device)
    bboxes_raw = result_od[0].boxes.xyxy
    bboxes = [Bbox(*bbox) for bbox in bboxes_raw]

    if len(bboxes) == 0:
        return ODOutput([], [])

    # Sort regions
    sorted_region_indices = sort_top_down_left_right(bboxes)
    sorted_bboxes   = [bboxes[i] for i in sorted_region_indices]
    sorted_polygons = [bbox_xyxy_to_polygon(bbox) for bbox in sorted_bboxes]
    return ODOutput(sorted_bboxes, sorted_polygons)


def line_seg(line_seg_model: YOLO, region_img: PILImage, device: str = "cpu") -> ODOutput:
    results_line_seg = line_seg_model(region_img, verbose=False, device=device)
    
    if results_line_seg[0].masks is None:
        return ODOutput([], [])

    # Sort masks
    masks               = [Polygon([(int(point[0]), int(point[1])) for point in mask]) for mask in results_line_seg[0].masks.xy]
    line_bboxes         = [Bbox(*polygon_to_bbox_xyxy(line)) for line in masks if len(line) > 0]
    sorted_indices      = sort_top_down_left_right(line_bboxes)
    sorted_bboxes       = [line_bboxes[i] for i in sorted_indices]
    sorted_polygons     = [masks[i] for i in sorted_indices]

    return ODOutput(sorted_bboxes, sorted_polygons)


# def line_od(line_od_model: YOLO, image: PILImage, device: str = "cpu") -> list[Bbox]:
#     results_line_od = line_od_model.predict(image, verbose=False, device=device)
#     line_bboxes_raw = results_line_od[0].boxes.xyxy
#     line_bboxes = [Bbox(*bbox) for bbox in line_bboxes_raw]

#     if len(line_bboxes) == 0:
#         return ODOutput([], [])

#     # Sort lines
#     sorted_indices = sort_top_down_left_right(line_bboxes)
#     sorted_line_bboxes = [line_bboxes[i] for i in sorted_indices]

#     return sorted_line_bboxes


def ocr(ocr_model: VisionEncoderDecoderModel, processor: TrOCRProcessor, line_images: list[PILImage], device: str = "cpu") -> list[str]:
    pixel_values    = processor(images=line_images, return_tensors="pt").pixel_values.to(device)
    generated_ids   = ocr_model.generate(inputs=pixel_values)
    line_trans      = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return line_trans
