import sys
from pathlib import Path
from abc import ABC, abstractmethod

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO
from htrflow.utils.geometry import Bbox
from shapely.geometry import Polygon
from PIL.Image import Image as PILImage
from tqdm import tqdm

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from src.data_processing.visual_tasks import polygon_to_bbox_xyxy, bbox_xyxy_to_polygon, crop_image, get_cover_bbox
from src.data_types import Page, Region, Line, ODOutput
from src.htr.utils import (
    sort_consider_margins,
    correct_line_bbox_coords,
    correct_line_polygon_coords,
    merge_overlapping_bboxes,
    sort_page
)
from src.logger import CustomLogger


REMOTE_TROCR_MODEL_PATH   = "microsoft/trocr-base-handwritten"


class Step(ABC):
    def __init__(self, model_path: str | Path, device: str = "cuda", logger: CustomLogger = None):
        self.model_path = model_path
        self.device = device

        if logger is not None:
            self.logger = logger
        else:
            self.logger = CustomLogger(self.__class__.__name__)

        self.model = self.load_model(model_path)
        self.processor = self.load_processor(REMOTE_TROCR_MODEL_PATH)

    def load_model(self, model_path: str | Path) -> YOLO:
        if "yolo" in str(model_path).lower():
            return YOLO(model_path)
        elif "trocr" in str(model_path).lower():
            return VisionEncoderDecoderModel.from_pretrained(model_path).to(self.device)

    def load_processor(self, processor_path: str | Path) -> TrOCRProcessor:
        return TrOCRProcessor.from_pretrained(processor_path)

    @abstractmethod
    def postprocess(self, image: PILImage, detected_objects: ODOutput) -> PILImage:
        pass

    @abstractmethod
    def run(self, image: PILImage) -> tuple[ODOutput, list[PILImage]]:
        pass


class TextObjectDetection(Step):
    """Capable of detecting regions or lines when given the right model"""
    def __init__(self, model_path: str | Path, device: str = "cuda", logger: CustomLogger = None):
        super().__init__(model_path, device, logger)

    def postprocess(self, image: PILImage, detected_objects: ODOutput) -> PILImage:
        cropped_imgs = []
        for polygon in detected_objects.polygons:
            cropped_imgs.append(crop_image(image, polygon))
        return cropped_imgs 
    
    def run(self, image: PILImage) -> tuple[ODOutput, list[PILImage]]:
        detected_objects    = self.detect(image)
        cropped_imgs        = self.postprocess(image, detected_objects)
        return detected_objects, cropped_imgs
    
    def detect(self, image: PILImage) -> ODOutput:
        try:
            result_od = self.model.predict(image, verbose=False, device=self.device)
        except Exception as e:
            self.logger.error(f"Cannot detect regions: {e}")
            return ODOutput(bboxes=[], polygons=[])
        
        bboxes_raw = result_od[0].boxes.xyxy
        bboxes = [Bbox(*bbox) for bbox in bboxes_raw]

        if len(bboxes) == 0:
            return ODOutput([], [])

        # Merge overlapping boxes
        merged_bboxes = merge_overlapping_bboxes(bboxes, iou_threshold=0.2, coverage_threshold=0.5)
        polygons = [bbox_xyxy_to_polygon(bbox) for bbox in merged_bboxes]
        return ODOutput(merged_bboxes, polygons)


class RegionTextSegmentation(Step):
    def __init__(self, model_path: str | Path, device: str = "cuda", logger: CustomLogger = None):
        super().__init__(model_path, device, logger)

    def postprocess(self, image: PILImage, detected_objects: ODOutput) -> PILImage:
        cropped_imgs = []
        for polygon in detected_objects.polygons:
            cropped_imgs.append(crop_image(image, polygon))
        return cropped_imgs 
    
    def run(self, image: PILImage) -> tuple[ODOutput, list[PILImage]]:
        detected_objects    = self.line_seg(image)
        cropped_imgs        = self.postprocess(image, detected_objects)
        return detected_objects, cropped_imgs
    
    def line_seg(self, image: PILImage) -> ODOutput:
        results_line_seg = self.model(image, verbose=False, device=self.device)
        
        if results_line_seg[0].masks is None:
            return ODOutput([], [])

        # Create output
        polygons = []
        bboxes = []
        for mask in results_line_seg[0].masks.xy:
            try:
                polygon = Polygon([(int(point[0]), int(point[1])) for point in mask])
                bbox    = Bbox(*polygon_to_bbox_xyxy(mask))
                polygons.append(polygon)
                bboxes.append(bbox)
            except Exception as e:
                print(e)
                continue

        return ODOutput(bboxes, polygons)


class SingleLineTextRecognition(Step):
    def __init__(self, model_path: str | Path, device: str = "cuda", logger: CustomLogger = None):
        super().__init__(model_path, device, logger)

    def postprocess(self):
        pass

    def run(self, images: list[PILImage]) -> list[str]:
        return self.ocr(images)

    def ocr(self, line_images: list[PILImage]) -> list[str]:
        pixel_values    = self.processor(images=line_images, return_tensors="pt").pixel_values.to(self.device)
        generated_ids   = self.model.generate(inputs=pixel_values)
        batch_texts     = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return batch_texts


class TraditionalPipeline():
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
            "region_od__line_seg__ocr": self.region_od__line_seg__ocr,
            "region_od__line_od__ocr": self.region_od__line_od__ocr,
            "line_od__ocr": self.line_od__ocr
        }

        assert pipeline_type in self.supported_pipelines, \
            f"pipeline_type must be one of {list(self.supported_pipelines.keys())}"

        self.pipeline_type           = pipeline_type
        self._region_od_model_path   = region_od_model_path
        self._line_od_model_path    = line_od_model_path
        self._line_seg_model_path    = line_seg_model_path
        self._ocr_model_path         = ocr_model_path

        self.device          = device
        self.batch_size      = batch_size

        if logger is None:
            self.logger = CustomLogger(f"pipeline__{pipeline_type}", log_to_local=True, log_path=PROJECT_DIR / "logs")
        else:
            self.logger = logger

        if "region_od" in pipeline_type:
            assert region_od_model_path is not None, "region_od_model_path must be provided if region_od is in pipeline_type"
            self.region_od = TextObjectDetection(self._region_od_model_path, self.device, self.logger)

        if "line_od" in pipeline_type:
            assert line_od_model_path is not None, "line_od_model_path must be provided if line_od is in pipeline_type"
            self.line_od = TextObjectDetection(self._line_od_model_path, self.device, self.logger)

        if "line_seg" in pipeline_type:
            assert line_seg_model_path is not None, "line_seg_model_path must be provided if line_seg is in pipeline_type"
            self.line_seg = RegionTextSegmentation(self._line_seg_model_path, self.device, self.logger)

        if "ocr" in pipeline_type:
            assert ocr_model_path is not None, "ocr_model_path must be provided if ocr is in pipeline_type"
            self.ocr = SingleLineTextRecognition(self._ocr_model_path, self.device, self.logger)


    def run(self, image: PILImage) -> Page:
        image = image.convert("RGB")
        return self.supported_pipelines[self.pipeline_type](image)

    
    def _region_od__line_det__ocr(self, image: PILImage, line_det_step: Step) -> Page:
        """Generic 3-step pipeline: region detection -> line detection -> ocr

        Parameters
        ----------
        image : PILImage
        line_det_step : Step
            pass in `self.line_od` or `self.line_seg` here

        Returns
        -------
        Page
        """

        ## Region detection
        self.logger.info("Region detection")
        page_region_od_output, page_region_imgs = self.region_od.run(image)
        page_region_lines = []

        for region_idx in range(len(page_region_od_output)):

            ## Line seg within region
            self.logger.info(f"Line detection for region {region_idx + 1}/{len(page_region_od_output)}")
            region_line_det_output, region_line_imgs = line_det_step.run(page_region_imgs[region_idx])
            
            ## OCR for lines within region
            self.logger.info(f"Text recognition for region {region_idx + 1}/{len(page_region_od_output)}")
            region_line_texts = []
            iterator = list(range(0, len(region_line_det_output), self.batch_size))

            for i in tqdm(iterator, total=len(iterator), unit="batch"):
                batch_indices = slice(i, i+self.batch_size)

                batch_texts = self.ocr.run(region_line_imgs[batch_indices])
                region_line_texts += batch_texts

            # Collect lines within regions
            assert len(region_line_det_output) == len(region_line_texts), "Length mismatch"

            # Shift line coords to match the larger page
            corrected_line_bboxes = [
                correct_line_bbox_coords(page_region_od_output.bboxes[region_idx], bbox)
                for bbox in region_line_det_output.bboxes
            ]

            corrected_line_polygons = [
                correct_line_polygon_coords(page_region_od_output.bboxes[region_idx], polygon)
                for polygon in region_line_det_output.polygons
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
    

    def region_od__line_od__ocr(self, image: PILImage) -> Page:
        return self._region_od__line_det__ocr(image, line_det_step=self.line_od)
    

    def region_od__line_seg__ocr(self, image: PILImage) -> Page:
        return self._region_od__line_det__ocr(image, line_det_step=self.line_seg)


    def line_od__ocr(self, image: PILImage) -> Page:

        ## Line detection
        self.logger.info("Line detection")
        page_line_od_output, page_line_imags = self.line_od.run(image)

        ## OCR
        self.logger.info("Batch text recognition")
        iterator = list(range(0, len(page_line_od_output.polygons), self.batch_size))
        page_line_texts = []

        for i in tqdm(iterator, total=len(iterator), unit="batch"):
            batch_indices = slice(i, i+self.batch_size)
            texts = self.ocr.run(page_line_imags[batch_indices])
            page_line_texts += texts

        # Output
        if page_line_od_output.bboxes == []:
            return Page(regions=[], lines=[])

        assert len(page_line_od_output.bboxes) == len(page_line_od_output.polygons) == len(page_line_texts), "Length mismatch"
        
        sorted_line_indices = sort_consider_margins(page_line_od_output.bboxes, image)
        
        # Output bbox, seg polygon created from bboxe, and text
        sorted_bboxes           = [page_line_od_output.bboxes[i] for i in sorted_line_indices]
        sorted_polygons         = [page_line_od_output.polygons[i] for i in sorted_line_indices]
        sorted_texts            = [page_line_texts[i] for i in sorted_line_indices]

        lines: list[Line] = [Line(*tup) for tup in zip(sorted_bboxes, sorted_polygons, sorted_texts)]

        # Get covering region. In line OD, there's only one region covering all line bboxes
        region_bbox = get_cover_bbox(sorted_bboxes)
        region_polygon = bbox_xyxy_to_polygon(region_bbox)
        region = Region(region_bbox, region_polygon, lines)

        page = sort_page(Page(regions=[region], lines=lines), image)
        return page