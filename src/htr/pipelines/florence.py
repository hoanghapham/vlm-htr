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
from src.data_processing.visual_tasks import bbox_xyxy_to_polygon, polygon_to_bbox_xyxy, crop_image, get_cover_bbox
from src.data_types import Page, Region, Line, ODOutput
from src.htr.utils import (
    sort_page,
    correct_line_bbox_coords,
    correct_line_polygon_coords,
    merge_overlapping_bboxes
)


os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

        bboxes          = [Bbox(*bbox) for bbox in bboxes_raw]
        merged_bboxes   = merge_overlapping_bboxes(bboxes, iou_threshold=0.2)
        polygons        = [bbox_xyxy_to_polygon(bbox) for bbox in merged_bboxes]
        return ODOutput(merged_bboxes, polygons)
    

class LineDetection(Step):
    def __init__(self, model_path: str | Path, device: str = "cuda", logger: CustomLogger = None):
        super().__init__(model_path, device, logger)

    def preprocess(self):
        pass

    def postprocess(self, image: PILImage, detected_objs: ODOutput) -> list[PILImage]:
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
        batch_size: int = 2,
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

    def run(self, image: PILImage) -> Page:
        image = image.convert("RGB")
        return self.supported_pipelines[self.pipeline_type](image)

    def line_od__line_seg__ocr(self, image: PILImage) -> Page:

        ## Line OD
        self.logger.info("Line detection")
        page_line_od_output, page_line_imgs = self.line_od.run(image)

        ## Line segmentation then OCR
        self.logger.info("Batch line segmentation -> Text recognition")
        page_line_texts = []
        page_line_segs = ODOutput(bboxes=[], polygons=[])
        iterator = list(range(0, len(page_line_od_output), self.batch_size))

        for i in tqdm(iterator, total=len(iterator), unit="batch"):
            batch_indices = slice(i, i+self.batch_size)

            # Text esegmentation within line image
            batch_seg_objs, batch_seg_imgs = self.line_seg.run(page_line_imgs[batch_indices])
            texts = self.ocr.run(batch_seg_imgs)
            
            page_line_segs += batch_seg_objs
            page_line_texts += texts

        # Output
        if page_line_od_output.bboxes == []:
            return Page(regions=[], lines=[])

        assert len(page_line_od_output) == len(page_line_segs) == len(page_line_texts), "Length mismatch"

        lines: list[Line] = [Line(*tup) for tup in zip(page_line_od_output.bboxes, page_line_segs.polygons, page_line_texts)]

        # Get covering region. In line OD, there's only one region covering all line bboxes
        region_bbox = get_cover_bbox(page_line_od_output.bboxes)
        region_polygon = bbox_xyxy_to_polygon(region_bbox)
        region = Region(region_bbox, region_polygon)

        page = sort_page(Page(regions=[region], lines=lines), image)
        return page


    def line_od__ocr(self, image: PILImage) -> Page:

        ## Line OD
        self.logger.info("Line detection")
        page_line_od_output, page_line_imgs = self.line_od.run(image)

        ## OCR
        self.logger.info("Batch text recognition")
        iterator = list(range(0, len(page_line_od_output.polygons), self.batch_size))
        page_line_texts = []

        for i in tqdm(iterator, total=len(iterator), unit="batch"):
            batch_indices = slice(i, i+self.batch_size)
            texts = self.ocr.run(page_line_imgs[batch_indices])
            page_line_texts += texts

        # Output

        if page_line_od_output.bboxes == []:
            return Page(regions=[], lines=[])

        assert len(page_line_od_output.bboxes) == len(page_line_od_output.polygons) == len(page_line_texts), "Length mismatch"

        # Assemble lines
        lines: list[Line] = [Line(*tup) for tup in zip(page_line_od_output.bboxes, page_line_od_output.polygons, page_line_texts)]

        region_bbox = get_cover_bbox(page_line_od_output.bboxes)
        region_polygon = bbox_xyxy_to_polygon(region_bbox)
        region = Region(region_bbox, region_polygon, lines)

        page = sort_page(Page(regions=[region], lines=lines), image)
        return page

    def region_od__line_od__ocr(self, image: PILImage) -> Page:
        ## Region detection
        self.logger.info("Region detection")
        page_region_od_output, page_region_imgs = self.region_od.run(image)
        page_region_lines = []

        for region_idx in range(len(page_region_od_output)):

            ## Line seg within region
            self.logger.info(f"Line detection for region {region_idx + 1}/{len(page_region_od_output)}")
            region_line_od_output, region_line_imgs = self.line_od.run(page_region_imgs[region_idx])
            
            ## OCR for lines within region
            self.logger.info(f"Text recognition for region {region_idx + 1}/{len(page_region_od_output)}")
            region_line_texts = []
            iterator = list(range(0, len(region_line_od_output), self.batch_size))

            for i in tqdm(iterator, total=len(iterator), unit="batch"):
                batch_indices = slice(i, i+self.batch_size)

                batch_texts = self.ocr.run(region_line_imgs[batch_indices])
                region_line_texts += batch_texts

            # Collect lines within regions
            assert len(region_line_od_output) == len(region_line_texts), "Length mismatch"

            # Shift line coords to match the larger page
            corrected_line_bboxes = [
                correct_line_bbox_coords(page_region_od_output.bboxes[region_idx], bbox)
                for bbox in region_line_od_output.bboxes
            ]

            corrected_line_polygons = [
                correct_line_polygon_coords(page_region_od_output.bboxes[region_idx], polygon)
                for polygon in region_line_od_output.polygons
            ]
            
            region_lines = [Line(*tup) for tup in zip(corrected_line_bboxes, corrected_line_polygons, region_line_texts)]
            page_region_lines.append(region_lines)

        # Collect regions within page
        page_regions = [Region(*tup) for tup in zip(page_region_od_output.bboxes, page_region_od_output.polygons, page_region_lines)]

        # Unnest region - lines
        page_lines = []
        for region_lines in page_region_lines:
            page_lines += region_lines

        # Final sorting
        page = sort_page(Page(regions=page_regions, lines=page_lines), image)
        return page
