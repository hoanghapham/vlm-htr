from htrflow.utils.geometry import Bbox
from src.data_processing.florence import predict, FlorenceTask
from src.post_process import topdown_left_right
from PIL.Image import Image as PILImage
from transformers import AutoModelForCausalLM, AutoProcessor


def region_od(region_od_model: AutoModelForCausalLM, processor: AutoProcessor, image: PILImage, device: str = "cpu") -> list[Bbox]:
    _, region_od_output = predict(
        region_od_model, 
        processor, 
        task_prompt=FlorenceTask.OD,
        user_prompt=None, 
        images=[image], 
        device=device
    )

    region_bboxes_raw = region_od_output[0][FlorenceTask.OD]["bboxes"]
    region_bboxes = [Bbox(*bbox) for bbox in region_bboxes_raw]

    # Sort regions
    sorted_region_indices = topdown_left_right(region_bboxes)
    sorted_region_bboxes = [region_bboxes[i] for i in sorted_region_indices]
    return sorted_region_bboxes


def line_od(line_od_model: AutoModelForCausalLM, processor: AutoProcessor, image: PILImage, device: str = "cpu") -> list[Bbox]:
    _, line_od_output = predict(
        line_od_model, 
        processor, 
        task_prompt=FlorenceTask.OD,
        user_prompt=None, 
        images=[image], 
        device=device
    )

    line_bboxes_raw = line_od_output[0][FlorenceTask.OD]["bboxes"]

    if len(line_bboxes_raw) == 0:
        return []

    line_bboxes = [Bbox(*bbox) for bbox in line_bboxes_raw]

    # Sort lines
    sorted_line_indices = topdown_left_right(line_bboxes)
    sorted_line_bboxes  = [line_bboxes[i] for i in sorted_line_indices]
    return sorted_line_bboxes
    

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