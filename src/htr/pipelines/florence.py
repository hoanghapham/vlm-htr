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
from src.htr.data_types import Page, Region, Line, ODOutput
from src.htr.utils import (
    sort_top_down_left_right, 
    sort_consider_margin,
    correct_line_bbox_coords,
    correct_line_polygon_coords
)


os.environ["TOKENIZERS_PARALLELISM"] = "false"



SORT_FUNCS = {
    "top_down_left_right": sort_top_down_left_right,
    "consider_margins": sort_consider_margin
}

MODEL_REMOTH_PATH = "microsoft/Florence-2-base-ft"
REVISION = 'refs/pr/6'



# Steps
class Step():
    def __init__(self, model_path: str | Path, device: str = "cuda", logger: CustomLogger = None):
        self.model_path = model_path
        self.device = device

        if logger is not None:
            self.logger = logger
        else:
            self.logger = CustomLogger(self.__class__.__name__)

        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained(MODEL_REMOTH_PATH, trust_remote_code=True, device_map=self.device)


class RegionDetection(Step):
    def __init__(self, model_path: str | Path, device: str = "cuda", logger: CustomLogger = None):
        super().__init__(model_path, device, logger)

    def preprocess(self):
        pass

    def postprocess(self, image: PILImage, detected_objs: ODOutput) -> PILImage:
        cropped_imgs = []
        for polygon in detected_objs.polygons:
            cropped_imgs.append(crop_image(image, polygon))
        return cropped_imgs

    def run(self, image: PILImage) -> tuple[ODOutput, list[PILImage]]:
        detected_objs   = self.detect(image)
        cropped_imgs    = self.postprocess(image, detected_objs)
        return detected_objs, cropped_imgs
    
    def detect(self, image: PILImage) -> ODOutput:
        """Perform region object detection on the page image"""
        try:
            _, output = predict(
                self.model, 
                self.processor, 
                task_prompt=FlorenceTask.OD,
                user_prompt=None, 
                images=[image], 
                device=self.device
            )
        except Exception as e:
            self.logger.error(f"Cannot detect regions: {e}")
            return ODOutput(bboxes=[], polygons=[])

        bboxes_raw = output[0][FlorenceTask.OD]["bboxes"]

        if len(bboxes_raw) == 0:
            return ODOutput(bboxes=[], polygons=[])

        bboxes      = [Bbox(*bbox) for bbox in bboxes_raw]
        polygons    = [bbox_xyxy_to_polygon(bbox) for bbox in bboxes]
        return ODOutput(bboxes, polygons)
    

class LineDetection(Step):
    def __init__(self, model_path: str | Path, device: str = "cuda", logger: CustomLogger = None):
        super().__init__(model_path, device, logger)

    def preprocess(self):
        pass

    def postprocess(self, image: PILImage, detected_objs: ODOutput) -> PILImage:
        cropped_imgs = []
        for polygon in detected_objs.polygons:
            cropped_imgs.append(crop_image(image, polygon))
        return cropped_imgs

    def run(self, image: PILImage) -> tuple[ODOutput, list[PILImage]]:
        detected_objs   = self.detect(image)
        cropped_imgs    = self.postprocess(image, detected_objs)
        return detected_objs, cropped_imgs

    def detect(self, image: PILImage) -> ODOutput:
        """Perform line object detection on the page image"""
        _, output = predict(
            self.model, 
            self.processor, 
            task_prompt=FlorenceTask.OD,
            user_prompt=None, 
            images=[image], 
            device=self.device
        )

        bboxes_raw = output[0][FlorenceTask.OD]["bboxes"]

        if len(bboxes_raw) == 0:
            return ODOutput(bboxes=[], polygons=[])

        bboxes      = [Bbox(*bbox) for bbox in bboxes_raw]
        polygons    = [bbox_xyxy_to_polygon(bbox) for bbox in bboxes]
        return ODOutput(bboxes, polygons)


class SingleLineTextSegmentation(Step):
    def __init__(self, model_path: str | Path, device: str = "cuda", logger: CustomLogger = None):
        super().__init__(model_path, device, logger)

    def preprocess(self):
        pass

    def postprocess(self, batch_line_imgs: list[PILImage], batch_seg_outputs: ODOutput) -> list[PILImage]:
        line_seg_imgs = []
        for line_img, polygon in zip(batch_line_imgs, batch_seg_outputs.polygons):
            line_seg_imgs.append(crop_image(line_img, polygon))
        return line_seg_imgs
    
    def run(self, images: list[PILImage]) -> tuple[ODOutput, list[PILImage]]:
        segmented_objects    = self.segment(images)
        line_seg_imgs        = self.postprocess(images, segmented_objects)
        return segmented_objects, line_seg_imgs

    def segment(self, images: list[PILImage]) -> ODOutput:
        """Perform line segmentation on a batch of rectangle line image"""
        try:
            _, output = predict(
                self.model, 
                self.processor, 
                task_prompt=FlorenceTask.REGION_TO_SEGMENTATION,
                user_prompt=None, 
                images=images, 
                device=self.device
            )
        except Exception as e:
            self.logger.warning(f"Failed to segment lines: {e}")
            return ODOutput(bboxes=[], polygons=[])

        raw_polygons   = [output[FlorenceTask.REGION_TO_SEGMENTATION]["polygons"][0][0] for output in output]

        if len(raw_polygons) == 0:
            return dict(bboxes=[], polygons=[])

        int_coords  = [[int(coord) for coord in mask] for mask in raw_polygons]
        polygons    = [Polygon(zip(mask[::2], mask[1::2])) for mask in int_coords]  # List of length 1
        bboxes      = [polygon_to_bbox_xyxy(poly) for poly in polygons]
        return ODOutput(bboxes, polygons)


class SingleLineTextRecognition(Step):
    def __init__(self, model_path: str | Path, device: str = "cuda", logger: CustomLogger = None):
        super().__init__(model_path, device, logger)

    def preprocess(self):
        pass
    
    def postprocess(self):
        pass

    def run(self, images: list[PILImage]) -> list[str]:
        return self.ocr(images)

    def ocr(self, line_images: list[PILImage]) -> list[str]:
        """Perform OCR on a batch of line images. Can be rectangle crop with or without segmentation mask
        Return a list of strings
        """
        try:
            _, ocr_output = predict(
                self.model, 
                self.processor, 
                task_prompt=FlorenceTask.OCR,
                user_prompt=None, 
                images=line_images, 
                device=self.device
            )
        except Exception as e:
            self.logger.warning(f"Failed to OCR lines: {e}")
            return [""] * len(line_images)

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
        
        self.supported_pipelines = {
            "region_od__line_od__ocr": self.region_od__line_od__ocr,
            "line_od__line_seg__ocr": self.line_od__line_seg__ocr,
            "line_od__ocr": self.line_od__ocr
        }

        assert pipeline_type in self.supported_pipelines, \
            f"pipeline_type must be one of {list(self.supported_pipelines.keys())}"

        self.pipeline_type           = pipeline_type
        self._region_od_model_path   = region_od_model_path
        self._line_od_model_path     = line_od_model_path
        self._line_seg_model_path    = line_seg_model_path
        self._ocr_model_path         = ocr_model_path
        self._remote_model_path      = "microsoft/Florence-2-base-ft"

        self.device          = device
        self.batch_size      = batch_size

        if logger is None:
            self.logger = CustomLogger(f"pipeline__{pipeline_type}", log_to_local=True, log_path=PROJECT_DIR / "logs")
        else:
            self.logger = logger
        
        if "region_od" in pipeline_type:
            assert region_od_model_path is not None, "region_od_model_path must be provided if region_od is in pipeline_type"
            self.region_od = RegionDetection(model_path=region_od_model_path, device=self.device, logger=self.logger)

        if "line_od" in pipeline_type:
            assert line_od_model_path is not None, "line_od_model_path must be provided if line_od is in pipeline_type"
            self.line_od = LineDetection(model_path=line_od_model_path, device=self.device, logger=self.logger)

        if "line_seg" in pipeline_type:
            assert line_seg_model_path is not None, "line_seg_model_path must be provided if line_seg is in pipeline_type"
            self.line_seg = SingleLineTextSegmentation(model_path=line_seg_model_path, device=self.device, logger=self.logger)

        if "ocr" in pipeline_type:
            assert ocr_model_path is not None, "ocr_model_path must be provided if ocr is in pipeline_type"
            self.ocr = SingleLineTextRecognition(model_path=ocr_model_path, device=self.device, logger=self.logger)

    def run(self, image: PILImage, sort_mode: str = "consider_margins") -> Page:
        return self.supported_pipelines[self.pipeline_type](image, sort_mode)

    def line_od__line_seg__ocr(self, image: PILImage, sort_mode: str = "consider_margins") -> Page:

        assert sort_mode in SORT_FUNCS.keys(), f"sort_mode must be one of {list(SORT_FUNCS.keys())}"

        self.logger.info("Line detection")
        page_line_objs, line_bbox_imgs = self.line_od.run(image)

        self.logger.info("Batch line segmentation -> Text recognition")
        page_line_texts = []
        page_line_segs = ODOutput(bboxes=[], polygons=[])
        iterator = list(range(0, len(page_line_objs), self.batch_size))

        for i in tqdm(iterator, total=len(iterator), unit="batch"):
            batch_indices = slice(i, i+self.batch_size)
            batch_seg_objs, batch_seg_imgs = self.line_seg.run(line_bbox_imgs[batch_indices])
            texts = self.ocr.run(batch_seg_imgs)
            
            page_line_segs += batch_seg_objs
            page_line_texts += texts

        # Sort output
        assert len(page_line_objs) == len(page_line_segs) == len(page_line_texts), "Length mismatch"
        if sort_mode == "top_down_left_right":
            sorted_line_indices = sort_top_down_left_right(page_line_objs.bboxes)
        elif sort_mode == "consider_margins":
            sorted_line_indices = sort_consider_margin(page_line_objs.bboxes, image)

        # Output bbox, seg polygon, and texts
        sorted_bboxes           = [page_line_objs.bboxes[i] for i in sorted_line_indices]
        sorted_polygons         = [page_line_segs.polygons[i] for i in sorted_line_indices]
        sorted_texts            = [page_line_texts[i] for i in sorted_line_indices]

        lines: list[Line] = [Line(*tup) for tup in zip(sorted_bboxes, sorted_polygons, sorted_texts)]
        return Page(regions=[], lines=lines)


    def line_od__ocr(self, image: PILImage, sort_mode: str = "consider_margins") -> Page:

        assert sort_mode in SORT_FUNCS.keys(), f"sort_mode must be one of {list(SORT_FUNCS.keys())}"

        self.logger.info("Line detection")
        line_od_output, line_bbox_imgs = self.line_od.run(image)

        ## Line seg then OCR
        self.logger.info("Batch text recognition")
        iterator = list(range(0, len(line_od_output.polygons), self.batch_size))
        page_line_texts = []

        for i in tqdm(iterator, total=len(iterator), unit="batch"):
            batch_indices = slice(i, i+self.batch_size)
            texts = self.ocr.run(line_bbox_imgs[batch_indices])
            page_line_texts += texts

        # Output
        assert len(line_od_output.bboxes) == len(line_od_output.polygons) == len(page_line_texts), "Length mismatch"
        
        if sort_mode == "top_down_left_right":
            sorted_line_indices = sort_top_down_left_right(line_od_output.bboxes)
        elif sort_mode == "consider_margins":
            sorted_line_indices = sort_consider_margin(line_od_output.bboxes, image)
        
        # Output bbox, seg polygon created from bboxe, and text
        sorted_bboxes           = [line_od_output.bboxes[i] for i in sorted_line_indices]
        sorted_polygons         = [line_od_output.polygons[i] for i in sorted_line_indices]
        sorted_texts            = [page_line_texts[i] for i in sorted_line_indices]

        lines: list[Line] = [Line(*tup) for tup in zip(sorted_bboxes, sorted_polygons, sorted_texts)]
        return Page(regions=[], lines=lines)
    

    def region_od__line_od__ocr(self, image: PILImage, sort_mode: str = "consider_margins") -> Page:
        
        assert sort_mode in SORT_FUNCS.keys(), f"sort_mode must be one of {list(SORT_FUNCS.keys())}"
        
        self.logger.info("Region detection")
        region_od_output, region_imgs = self.region_od.run(image)
        
        ## Line OD within region
        self.logger.info("Line detection within region")
        page_regions: list[tuple[ODOutput, list[str]]] = []

        for region_idx in range(len(region_od_output.bboxes)):

            ## Line OD
            self.logger.info("Line detection")
            region_line_objs, line_bbox_imgs = self.line_od.run(region_imgs[region_idx])

            ## OCR
            self.logger.info("Text recognition")
            iterator = list(range(0, len(region_line_objs.polygons), self.batch_size))
            region_line_texts = []
            
            for i in tqdm(iterator, total=len(iterator), unit="batch", desc=f"OCR for region {region_idx + 1}/{len(region_od_output)}"):

                batch_indices = slice(i, i+self.batch_size)
                texts = self.ocr.run(line_bbox_imgs[batch_indices])
                region_line_texts += texts
            
            # Gather data for one region
            page_regions.append((region_line_objs, region_line_texts))
        
        # Final sorting step
        # Sort regions
        assert len(page_regions) == len(region_od_output), \
            f"Length mismatch: {len(page_regions)} - {len(region_od_output)}"
        
        if sort_mode == "top_down_left_right":
            sorted_region_indices = sort_top_down_left_right(region_od_output.bboxes)
        elif sort_mode == "consider_margins":
            sorted_region_indices = sort_consider_margin(region_od_output.bboxes, image)

        sorted_region_bboxes    = [region_od_output.bboxes[i] for i in sorted_region_indices]
        sorted_region_polygons  = [region_od_output.polygons[i] for i in sorted_region_indices]
        sorted_region_lines     = []

        # Get region lines
        for region_idx in sorted_region_indices:
            region_line_objs, region_line_texts = page_regions[region_idx]

            if sort_mode == "top_down_left_right":
                sorted_line_indices = sort_top_down_left_right(region_line_objs.bboxes)
            elif sort_mode == "consider_margins":
                sorted_line_indices = sort_consider_margin(region_line_objs.bboxes, image)

            corrected_line_bboxes = []
            for bbox in region_line_objs.bboxes:
                corrected_bbox = correct_line_bbox_coords(
                    region_od_output.bboxes[region_idx], 
                    bbox
                )
                corrected_line_bboxes.append(corrected_bbox)
            
            corrected_line_polygons = []
            for polygon in region_line_objs.polygons:
                corrected_polygon = correct_line_polygon_coords(
                    region_od_output.bboxes[region_idx], 
                    polygon
                )
                corrected_line_polygons.append(corrected_polygon)

            sorted_line_bboxes      = [corrected_line_bboxes[i] for i in sorted_line_indices]
            sorted_line_polygons    = [corrected_line_polygons[i] for i in sorted_line_indices]
            sorted_line_texts       = [region_line_texts[i] for i in sorted_line_indices]

            region_lines = [Line(*tup) for tup in zip(sorted_line_bboxes, sorted_line_polygons, sorted_line_texts)]
            sorted_region_lines.append(region_lines)
        
        page_regions = [Region(*tup) for tup in zip(sorted_region_bboxes, sorted_region_polygons, sorted_region_lines)]

        page_lines = []
        for lines in sorted_region_lines:
            page_lines += lines

        return Page(regions=page_regions, lines=page_lines)
