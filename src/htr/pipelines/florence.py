#%%
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL.Image import Image as PILImage
from htrflow.utils.geometry import Bbox
from shapely.geometry import Polygon

from src.logger import CustomLogger
from src.data_processing.florence import predict, FlorenceTask
from src.data_processing.visual_tasks import bbox_xyxy_to_polygon, polygon_to_bbox_xyxy, crop_image
from src.htr.postprocess import sort_top_down_left_right, sort_consider_margin
from src.htr.data_types import Page, Region, Line, ODOutput


os.environ["TOKENIZERS_PARALLELISM"] = "false"


SUPPORTED_PIPELINES = [
    "region_od__line_od__ocr",
    "line_od__line_seg__ocr",
    "line_od__ocr"
]

sort_funcs = {
    "top_down_left_right": sort_top_down_left_right,
    "consider_margins": sort_consider_margin
}


# Steps
def region_od(region_od_model: AutoModelForCausalLM, processor: AutoProcessor, image: PILImage, device: str = "cpu") -> ODOutput:
    """Perform region object detection on the page image"""
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
        return ODOutput(bboxes=[], polygons=[])

    # Sort regions
    bboxes           = [Bbox(*bbox) for bbox in bboxes_raw]
    # sorted_indices   = sort_top_down_left_right(bboxes)
    # sorted_indices   = sort_bboxes(image, bboxes)
    # sorted_bboxes    = [bboxes[i] for i in sorted_indices]
    polygons  = [bbox_xyxy_to_polygon(bbox) for bbox in bboxes]
    return ODOutput(bboxes, polygons)


def line_od(line_od_model: AutoModelForCausalLM, processor: AutoProcessor, image: PILImage, device: str = "cpu") -> ODOutput:
    """Perform line object detection on the page image"""
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
        return ODOutput(bboxes=[], polygons=[])

    # Sort lines
    bboxes              = [Bbox(*bbox) for bbox in bboxes_raw]
    # sorted_indices      = sort_top_down_left_right(bboxes)
    # sorted_indices      = sort_bboxes(image, bboxes)
    # sorted_bboxes       = [bboxes[i] for i in sorted_indices]
    polygons     = [bbox_xyxy_to_polygon(bbox) for bbox in bboxes]
    return ODOutput(bboxes, polygons)


def line_seg(line_seg_model: AutoModelForCausalLM, processor: AutoProcessor, line_image: list[PILImage], device: str = "cpu") -> ODOutput:
    """Perform line segmentation on a rectangle line image"""
    _, output = predict(
        line_seg_model, 
        processor, 
        task_prompt=FlorenceTask.REGION_TO_SEGMENTATION,
        user_prompt=None, 
        images=line_image, 
        device=device
    )

    raw_polygons   = [output[FlorenceTask.REGION_TO_SEGMENTATION]["polygons"][0][0] for output in output]

    if len(raw_polygons) == 0:
        return dict(bboxes=[], polygons=[])

    int_coords  = [[int(coord) for coord in mask] for mask in raw_polygons]
    polygons    = [Polygon(zip(mask[::2], mask[1::2])) for mask in int_coords]  # List of length 1
    bboxes      = [polygon_to_bbox_xyxy(poly) for poly in polygons]
    return ODOutput(bboxes, polygons)


def ocr(ocr_model: AutoModelForCausalLM, processor: AutoProcessor, line_images: list[PILImage], device: str = "cpu") -> list[str]:
    """Perform OCR on a line image. Can be rectangle crop with or without segmentation mask
    Return a list of strings
    """
    _, ocr_output = predict(
        ocr_model, 
        processor, 
        task_prompt=FlorenceTask.OCR,
        user_prompt=None, 
        images=line_images, 
        device=device
    )

    batch_texts = [output[FlorenceTask.OCR].replace("<pad>", "") for output in ocr_output]
    return batch_texts


# Pipelines

class FlorencePipeline():
    def __init__(
        self, 
        pipeline_type: str,
        region_od_model_path: str | Path = None, 
        line_od_model_path: str | Path = None, 
        line_seg_model_path: str | Path = None, 
        ocr_model_path: str | Path = None, 
        batch_size: str = 2,
        device: str = "cuda",
        logger: CustomLogger = None,
    ):
        
        assert pipeline_type in SUPPORTED_PIPELINES, f"pipeline_type must be one of {SUPPORTED_PIPELINES}"

        self.pipeline_type           = pipeline_type
        self._region_od_model_path   = region_od_model_path
        self._line_od_model_path     = line_od_model_path
        self._line_seg_model_path    = line_seg_model_path
        self._ocr_model_path         = ocr_model_path
        self._remote_model_path      = "microsoft/Florence-2-base-ft"

        self.device          = device
        self.batch_size      = batch_size
        self.region_od_model = None
        self.line_od_model   = None
        self.line_seg_model  = None 
        self.ocr_model       = None
        self.processor       = AutoProcessor.from_pretrained(
            self._remote_model_path, trust_remote_code=True, device_map=self.device)

        if "region_od" in pipeline_type:
            self.region_od_model = self._load_model(self._region_od_model_path)

        if "line_od" in pipeline_type:
            self.line_od_model = self._load_model(self._line_od_model_path)

        if "line_seg" in pipeline_type:
            self.line_seg_model = self._load_model(self._line_seg_model_path)

        if "ocr" in pipeline_type:
            self.ocr_model = self._load_model(self._ocr_model_path)

        if logger is None:
            self.logger = CustomLogger(f"pipeline__{pipeline_type}", log_to_local=True, log_path=PROJECT_DIR / "logs")
        else:
            self.logger = logger

    def _load_model(self, model_path):
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(self.device)
        return model
        

    def line_od__line_seg__ocr(self, image: PILImage, sort_mode: str = "top_down_left_right") -> Page:

        assert sort_mode in sort_funcs.keys(), f"sort_mode must be one of {list(sort_funcs.keys())}"

        self.logger.info("Line detection")
        try:
            line_od_output = line_od(self.line_od_model, self.processor, image, self.device)
        except Exception as e:
            self.logger.warning(f"Failed to detect lines: {e}")
            return Page([], [])

        if len(line_od_output.bboxes) == 0:
            self.logger.warning("Can't detect lines on the page")
            return Page([], [])

        ## Line seg then OCR
        self.logger.info("Line segmentation -> Text recognition")
        iterator = list(range(0, len(line_od_output.polygons), self.batch_size))
        page_line_texts = []

        for i in tqdm(iterator, total=len(iterator), unit="batch"):

            batch = line_od_output.polygons[i:i + self.batch_size]
            batch_line_bbox_imgs = []

            for bbox_polygon in batch:
                batch_line_bbox_imgs.append(crop_image(image, bbox_polygon))

            # Line segmentation
            try:
                line_seg_output = line_seg(self.line_seg_model, self.processor, batch_line_bbox_imgs, self.device)
            except Exception as e:
                self.logger.warning(f"Failed to segment lines: {e}")
                continue

            batch_line_seg_imgs = []

            for line_img, mask in zip(batch_line_bbox_imgs, line_seg_output.polygons):
                batch_line_seg_imgs.append(crop_image(line_img, mask))
            
            # OCR
            try:
                batch_texts = ocr(self.ocr_model, self.processor, batch_line_seg_imgs, self.device)
            except Exception as e:
                self.logger.warning(f"Failed to OCR line images: {e}")
                batch_texts = [""] * len(self.batch_size)
                continue

            page_line_texts += batch_texts

        # Sort output
        assert len(line_od_output.bboxes) == len(line_od_output.polygons) == len(page_line_texts), "Length mismatch"
        if sort_mode == "top_down_left_right":
            sorted_line_indices = sort_top_down_left_right(line_od_output.bboxes)
        elif sort_mode == "consider_margins":
            sorted_line_indices = sort_consider_margin(line_od_output.bboxes, image)

        sorted_bboxes           = [line_od_output.bboxes[i] for i in sorted_line_indices]
        sorted_polygons         = [line_od_output.polygons[i] for i in sorted_line_indices]
        sorted_texts            = [page_line_texts[i] for i in sorted_line_indices]

        lines: list[Line] = [Line(*tup) for tup in zip(sorted_bboxes, sorted_polygons, sorted_texts)]
        return Page(regions=[], lines=lines)
    

    def line_od__ocr(self, image: PILImage, sort_mode: str = "top_down_left_right") -> Page:

        assert sort_mode in sort_funcs.keys(), f"sort_mode must be one of {list(sort_funcs.keys())}"

        self.logger.info("Line detection")
        try:
            line_od_output = line_od(self.line_od_model, self.processor, image, self.device)
        except Exception as e:
            self.logger.warning(f"Failed to detect lines: {e}")
            return Page([], [])

        if len(line_od_output.bboxes) == 0:
            self.logger.warning("Can't detect lines on the page")
            return Page([], [])

        ## Line seg then OCR
        self.logger.info("Line segmentation -> Text recognition")
        iterator = list(range(0, len(line_od_output.polygons), self.batch_size))
        page_line_texts = []

        for i in tqdm(iterator, total=len(iterator), unit="batch"):

            # Create a batch of cropped line bboxes
            # Cut line segs from region images
            batch = line_od_output.polygons[i:i+self.batch_size]
            cropped_line_imgs = []
            
            for bbox_coords in batch:
                line_img = crop_image(image, bbox_coords)
                cropped_line_imgs.append(line_img)

            # Batch inference
            try:
                batch_texts = ocr(self.ocr_model, self.processor, cropped_line_imgs, self.device)
            except Exception as e:
                self.logger.warning(f"Failed to OCR line images: {e}")
                batch_texts = [""] * len(self.batch_size)
                continue

            page_line_texts += batch_texts

        # Output
        assert len(line_od_output.bboxes) == len(line_od_output.polygons) == len(page_line_texts), "Length mismatch"
        
        if sort_mode == "top_down_left_right":
            sorted_line_indices = sort_top_down_left_right(line_od_output.bboxes)
        elif sort_mode == "consider_margins":
            sorted_line_indices = sort_consider_margin(line_od_output.bboxes, image)
        
        sorted_bboxes           = [line_od_output.bboxes[i] for i in sorted_line_indices]
        sorted_polygons         = [line_od_output.polygons[i] for i in sorted_line_indices]
        sorted_texts            = [page_line_texts[i] for i in sorted_line_indices]

        lines: list[Line] = [Line(*tup) for tup in zip(sorted_bboxes, sorted_polygons, sorted_texts)]
        return Page(regions=[], lines=lines)
    

    def region_od__line_od__ocr(self, image: PILImage, sort_mode: str = "top_down_left_right") -> Page:
        
        assert sort_mode in sort_funcs.keys(), f"sort_mode must be one of {list(sort_funcs.keys())}"
        
        self.logger.info("Region detection")
        try:
            region_od_output = region_od(self.region_od_model, self.processor, image, self.device)
        except Exception as e:
            self.logger.warning(f"Failed to detect regions: {e}")
            return Page([], [])

        if len(region_od_output.bboxes) == 0:
            self.logger.warning("Can't detect regions on the page")
            return Page([], [])
        
        ## Line OD within region
        self.logger.info("Line detection within region")
        regions_line_objects: list[(ODOutput, list[str])] = []

        for region_polygon in region_od_output.polygons:

            ## Line OD
            try:
                region_img  = crop_image(image, region_polygon)
                line_od_output = line_od(self.line_od_model, self.processor, region_img, self.device)
            except Exception as e:
                self.logger.warning(f"Failed to detect lines: {e}")
                regions_line_objects.append((ODOutput([], []), []))
                continue
            
            if len(line_od_output.bboxes) == 0:
                self.logger.warning("Can't find lines on the region image")
                regions_line_objects.append((ODOutput([], []), []))
                continue

            ## OCR
            self.logger.info("Text recognition")
            iterator = list(range(0, len(line_od_output.polygons), self.batch_size))
            line_texts = []
            
            for i in tqdm(iterator, total=len(iterator), unit="batch"):

                # Create a batch of cropped line bboxes
                # Aggregate line images to do batch OCR
                batch = line_od_output.polygons[i:i+self.batch_size]
                batch_line_imgs = []

                for mask in batch:
                    line_img = crop_image(image, mask)
                    batch_line_imgs.append(line_img)

                # OCR
                try:
                    batch_texts = ocr(self.ocr_model, self.processor, batch_line_imgs, self.device)
                    line_texts += batch_texts
                except Exception as e:
                    self.logger.warning(f"Failed to OCR line images: {e}")
                    batch_texts = [""] * len(self.batch_size)
                    continue
            
            # Gather data for one region
            regions_line_objects.append((line_od_output, line_texts))
        
        # Final sorting step
        # Sort regions
        assert len(regions_line_objects) == len(region_od_output), \
            f"Length mismatch: {len(regions_line_objects)} - {len(region_od_output)}"
        
        if sort_mode == "top_down_left_right":
            sorted_region_indices = sort_top_down_left_right(region_od_output.bboxes)
        elif sort_mode == "consider_margins":
            sorted_region_indices = sort_consider_margin(region_od_output.bboxes, image)
        sorted_region_bboxes    = [region_od_output.bboxes[i] for i in sorted_region_indices]
        sorted_region_polygons  = [region_od_output.polygons[i] for i in sorted_region_indices]
        sorted_region_lines     = []

        # Get region lines
        for region_idx in sorted_region_indices:

            region_line_objs, region_line_texts = regions_line_objects[region_idx]

            if sort_mode == "top_down_left_right":
                sorted_line_indices = sort_top_down_left_right(region_line_objs.bboxes)
            elif sort_mode == "consider_margins":
                sorted_line_indices = sort_consider_margin(region_line_objs.bboxes, image)

            sorted_line_bboxes      = [region_line_objs.bboxes[i] for i in sorted_line_indices]
            sorted_line_polygons    = [region_line_objs.polygons[i] for i in sorted_line_indices]
            sorted_line_texts       = [region_line_texts[i] for i in sorted_line_indices]

            region_lines = [Line(*tup) for tup in zip(sorted_line_bboxes, sorted_line_polygons, sorted_line_texts)]
            sorted_region_lines.append(region_lines)
        
        page_regions = [Region(*tup) for tup in zip(sorted_region_bboxes, sorted_region_polygons, sorted_region_lines)]

        page_lines = []
        for lines in sorted_region_lines:
            page_lines += lines

        return Page(regions=page_regions, lines=page_lines)
