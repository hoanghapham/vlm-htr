#%%
import sys
from pathlib import Path

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))
from PIL.Image import Image as PILImage

from src.data_processing.visual_tasks import crop_image
from src.logger import CustomLogger
from src.htr.steps.florence import line_od, line_seg, ocr
from src.htr.output_types import Line, Page

SUPPORTED_PIPELINES = [
    "region_od__line_od__ocr",
    "line_od__line_seg__ocr",
    "line_od__ocr"
]


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
        
    def line_od__line_seg__ocr(self, image: PILImage):

        self.logger.info("Line detection")
        line_od_output = line_od(self.line_od_model, self.processor, image, self.device)

        ## Line seg then OCR
        self.logger.info("Line segmentation -> Text recognition")
        lines_text = []

        # Iterate through detected lines
        iterator = list(range(0, len(line_od_output.polygons), self.batch_size))

        for i in tqdm(iterator, total=len(iterator), unit="batch"):

            # Create a batch of cropped line bboxes
            batch = line_od_output.polygons[i:i + self.batch_size]
            batch_line_bbox_imgs = []

            for bbox_polygon in batch:
                batch_line_bbox_imgs.append(crop_image(image, bbox_polygon))

            # Line segmentation
            line_seg_output     = line_seg(self.line_seg_model, self.processor, batch_line_bbox_imgs, self.device)
            batch_line_seg_imgs = []

            for line_img, mask in zip(batch_line_bbox_imgs, line_seg_output.polygons):
                batch_line_seg_imgs.append(crop_image(line_img, mask))
            
            # OCR
            batch_text = ocr(self.ocr_model, self.processor, batch_line_seg_imgs, self.device)
            lines_text += batch_text

        # Output
        lines: list[Line] = [Line(*tup) for tup in zip(line_od_output.bboxes, line_od_output.polygons, lines_text)]
        return Page(regions=[], lines=lines)
