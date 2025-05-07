import sys
import os
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoProcessor
from htrflow.utils.geometry import Bbox
from shapely.geometry import Polygon
from PIL.Image import Image as PILImage

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from src.data_processing.florence import predict, FlorenceTask
from src.data_processing.visual_tasks import bbox_xyxy_to_polygon, polygon_to_bbox_xyxy
from src.htr.steps.postprocess import sort_top_down_left_right, sort_bboxes

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ODOutput():
    def __init__(self, object_bboxes: list[Bbox], object_polygons: list[Polygon]):
        self.bboxes = object_bboxes
        self.polygons = object_polygons


def region_od(region_od_model: AutoModelForCausalLM, processor: AutoProcessor, image: PILImage, device: str = "cpu") -> ODOutput:
    _, output = predict(
        region_od_model, 
        processor, 
        task_prompt=FlorenceTask.OD,
        user_prompt=None, 
        images=[image], 
        device=device
    )

    bboxes_raw = output[0][FlorenceTask.OD]["bboxes"]

    if len(bboxes_raw) == 0:
        return ODOutput([], [])

    # Sort regions
    bboxes           = [Bbox(*bbox) for bbox in bboxes_raw]
    # sorted_indices   = sort_top_down_left_right(bboxes)
    sorted_indices   = sort_bboxes(image, bboxes)
    sorted_bboxes    = [bboxes[i] for i in sorted_indices]
    sorted_polygons  = [bbox_xyxy_to_polygon(bbox) for bbox in sorted_bboxes]

    return ODOutput(sorted_bboxes, sorted_polygons)


def line_od(line_od_model: AutoModelForCausalLM, processor: AutoProcessor, image: PILImage, device: str = "cpu") -> ODOutput:
    _, output = predict(
        line_od_model, 
        processor, 
        task_prompt=FlorenceTask.OD,
        user_prompt=None, 
        images=[image], 
        device=device
    )

    bboxes_raw = output[0][FlorenceTask.OD]["bboxes"]

    if len(bboxes_raw) == 0:
        return ODOutput([], [])

    # Sort lines
    bboxes              = [Bbox(*bbox) for bbox in bboxes_raw]
    # sorted_indices      = sort_top_down_left_right(bboxes)
    sorted_indices      = sort_bboxes(image, bboxes)
    sorted_bboxes       = [bboxes[i] for i in sorted_indices]
    sorted_polygons     = [bbox_xyxy_to_polygon(bbox) for bbox in sorted_bboxes]
    return ODOutput(sorted_bboxes, sorted_polygons)
    

def line_seg(line_seg_model: AutoModelForCausalLM, processor: AutoProcessor, cropped_line_images: list[PILImage], device: str = "cpu") -> ODOutput:
    _, output = predict(
        line_seg_model, 
        processor, 
        task_prompt=FlorenceTask.REGION_TO_SEGMENTATION,
        user_prompt=None, 
        images=cropped_line_images, 
        device=device
    )

    raw_polygons   = [output[FlorenceTask.REGION_TO_SEGMENTATION]["polygons"][0][0] for output in output]

    if len(raw_polygons) == 0:
        return ODOutput([], [])

    int_coords  = [[int(coord) for coord in mask] for mask in raw_polygons]
    polygons    = [Polygon(zip(mask[::2], mask[1::2])) for mask in int_coords]  # List of length 1
    bboxes      = [polygon_to_bbox_xyxy(poly) for poly in polygons]
    return ODOutput(bboxes, polygons)


def ocr(ocr_model: AutoModelForCausalLM, processor: AutoProcessor, line_images: list[PILImage], device: str = "cpu") -> list[str]:
    _, ocr_output = predict(
        ocr_model, 
        processor, 
        task_prompt=FlorenceTask.OCR,
        user_prompt=None, 
        images=line_images, 
        device=device
    )

    line_trans = [output[FlorenceTask.OCR].replace("<pad>", "") for output in ocr_output]
    return line_trans