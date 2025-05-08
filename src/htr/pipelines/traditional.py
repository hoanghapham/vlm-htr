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

from src.data_processing.visual_tasks import polygon_to_bbox_xyxy, bbox_xyxy_to_polygon, crop_image
from src.htr.utils import sort_top_down_left_right, sort_consider_margin
from src.htr.data_types import Page, Region, Line, ODOutput
from src.logger import CustomLogger

SUPPORTED_PIPELINES = [
    "region_od__line_od__ocr",
    "region_od__line_seg__ocr",
    "line_od__ocr"
]

SORT_FUNCS = {
    "top_down_left_right": sort_top_down_left_right,
    "consider_margins": sort_consider_margin
}

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

        # Sort regions
        polygons = [bbox_xyxy_to_polygon(bbox) for bbox in bboxes]
        return ODOutput(bboxes, polygons)


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

    def ocr(self, line_images: list[PILImage], device: str = "cpu") -> list[str]:
        pixel_values    = self.processor(images=line_images, return_tensors="pt").pixel_values.to(device)
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
        logger: CustomLogger = None
    ):
        assert pipeline_type in SUPPORTED_PIPELINES, f"pipeline_type must be one of {SUPPORTED_PIPELINES}"

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

    
    def region_od__line_seg__ocr(self, image: PILImage, sort_mode: str = "top_down_left_right") -> Page:
        assert sort_mode in SORT_FUNCS.keys(), f"sort_mode must be one of {list(SORT_FUNCS.keys())}"

        self.logger.info("Region detection")
        region_od_output, region_imgs = self.region_od.run(image)

        page_regions: list[tuple[ODOutput, list[str]]] = []

        for region_idx in range(len(region_od_output)):
            self.logger.info(f"Line segmentation for region {region_idx}")
            region_line_objs, region_line_imgs = self.line_seg.run(region_imgs[region_idx])
            
            iterator = list(range(0, len(region_line_objs), self.batch_size))

            region_line_texts = []
            for i in tqdm(iterator, total=len(iterator), unit="batch", desc=f"OCR for region {region_idx}/{len(region_od_output)}"):
                batch_indices = slice(i, i+self.batch_size)
                batch_texts = self.ocr.run(region_line_imgs[batch_indices])
                region_line_texts += batch_texts

            # Collect data for one region
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