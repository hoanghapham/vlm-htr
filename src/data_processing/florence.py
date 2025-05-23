import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from typing import Sequence
import torch
from torch.utils.data import Dataset
from PIL import Image
from PIL.Image import Image as PILImage

from src.data_processing.utils import load_arrow_datasets
from src.data_processing.visual_tasks import crop_image, bbox_xyxy_to_polygon
from src.data_processing.base_datasets import BaseImgXMLDataset


class FlorenceTask():
    OD = "<OD>"
    DENSE_REGION_CAPTION = "<DENSE_REGION_CAPTION>"
    CAPTION = "<CAPTION>"
    DETAILED_CAPTION = "<DETAILED_CAPTION>"
    MORE_DETAILED_CAPTION = "<MORE_DETAILED_CAPTION>"
    REGION_PROPOSAL = "<REGION_PROPOSAL>"
    CAPTION_TO_PHRASE_GROUNDING = "<CAPTION_TO_PHRASE_GROUNDING>"
    REFERRING_EXPRESSION_SEGMENTATION = "<REFERRING_EXPRESSION_SEGMENTATION>"
    REGION_TO_SEGMENTATION = "<REGION_TO_SEGMENTATION>"
    OPEN_VOCABULARY_DETECTION = "<OPEN_VOCABULARY_DETECTION>"
    REGION_TO_DESCRIPTION = "<REGION_TO_DESCRIPTION>"
    OCR = "<OCR>"
    OCR_WITH_REGION = "<OCR_WITH_REGION>"


def extract_florence_seg_polygon(task, parsed_result):
    segm = parsed_result[task]["polygons"][0][0]
    seg_x = segm[0::2]
    seg_y = segm[1::2]
    polygon = list(zip(seg_x, seg_y))
    return polygon


def bbox_xyxy_to_florence(bbox, box_quantizer, image):
    quantized_bbox = box_quantizer.quantize(torch.Tensor(bbox), size=image.size)
    text = "".join([f"<loc_{val}>" for val in quantized_bbox])
    return text


def bboxes_xyxy_to_florence(bbox, box_quantizer, image):
    quantized_bboxes = box_quantizer.quantize(torch.Tensor([bbox]), size=image.size)
    bbox_texts = []
    for bbox in quantized_bboxes:
        text = "".join([f"<loc_{val}>" for val in bbox])
        bbox_texts.append(text)
    
    return bbox_texts


def coords_to_florence(coords: list[tuple], coords_quantizer, image):
    """Receive a list of coord tuples [(x1, y1), (x2, y2), ...] and convert to Florence string format"""
    quant_poly = coords_quantizer.quantize(torch.Tensor(coords), size=image.size)
    points_str = ""
    for point in quant_poly:
        points_str += f"<loc_{point[0]}><loc_{point[1]}>"
    return points_str


def polygons_to_florence(polygons: list[list[tuple]], coords_quantizer, image):
    polygon_texts = []
    for polygon in polygons:
        points_str = coords_to_florence(polygon, coords_quantizer, image)
        polygon_texts.append(points_str)
    return polygon_texts


class FlorenceOCRDataset(Dataset):
    """
    Load locally cached .arrow dataset containing cropped lines/regions
    """

    def __init__(self, dir_path: str | Path, custom_question: str = None):
        self.data = load_arrow_datasets(dir_path)

        if custom_question:
            self.question = custom_question
        else:
            self.question = FlorenceTask.OCR

    def __len__(self):
        return len(self.data)

    def _get_one(self, idx):
        example = self.data[idx]
        question = self.question
        answer = example["transcription"]
        image = example['image'].convert("RGB")
        return dict(
            question=question, 
            answer=answer, 
            image=image
        )
    
    def __getitem__(self, index: int | slice):
        if isinstance(index, int):
            return self._get_one(index)
        else:
            return [self._get_one(idx) for idx in range(index.start, index.stop, index.step or 1) if idx < len(self.data)]


class FlorenceSingleLineSegDataset(BaseImgXMLDataset):
    """Dataset that returns one rectangular crop of a line, with polygon seg mask"""

    def __init__(self, data_dir: str | Path):
        super().__init__(data_dir)

        self.task               = FlorenceTask.REGION_TO_SEGMENTATION
        self.box_quantizer      = BoxQuantizer("floor", (1000, 1000))
        self.coords_quantizer   = CoordinatesQuantizer("floor", (1000, 1000))

        # Validate that the xml file has the data type we need, then set self.img_paths and self.xml_paths
        self.img_paths = []
        self.xml_paths = []
        self.img_paths, self.xml_paths = self.validate_and_load(self._all_img_paths, self._all_xml_paths)

    def validate_and_load(self, img_paths, xml_paths):
        valid_img_paths = []
        valid_xml_paths = []

        # List to convert a global line idx to path of an image
        self.line_to_img_path = []
        
        # Pre-load lines data from all XMLs
        self.lines_data = []

        for idx, xml in enumerate(xml_paths):
            lines = self.xmlparser.get_lines(xml) # Fields: region_id, line_id, bbox, polygon, transcription
            
            if len(lines) > 0:
                valid_img_paths.append(img_paths[idx])
                valid_xml_paths.append(xml)

            self.lines_data += lines
            self.line_to_img_path += [img_paths[idx]] * len(lines)
        
        return valid_img_paths, valid_xml_paths

    @property
    def nsamples(self):
        return len(self.lines_data)

    def _get_one(self, idx):
        image       = Image.open(self.line_to_img_path[idx]).convert("RGB")
        data        = self.lines_data[idx]
        unique_key  = data["unique_key"]

        # Crop image to a line using bbox
        bbox_coords = bbox_xyxy_to_polygon(data["bbox"])
        cropped_line_img = crop_image(image, bbox_coords)
        
        # Shift bbox and polygon to follow the newly cropped images
        shift_x = data["bbox"][0]
        shift_y = data["bbox"][1]

        shifted_bbox = (
            0, 
            0,
            data["bbox"][2] - shift_x, 
            data["bbox"][3] - shift_y
        )

        new_polygon = [(x - shift_x, y - shift_y) for (x, y) in data["polygon"]]

        # Convert bbox and polygon to florence text format
        florence_bbox   = bbox_xyxy_to_florence(shifted_bbox, self.box_quantizer, cropped_line_img)
        florence_polygon = coords_to_florence(new_polygon, self.coords_quantizer, cropped_line_img)

        # Form input question
        question = self.task + florence_bbox

        return dict(
            unique_key = unique_key,
            image = cropped_line_img,
            question = question,
            answer = florence_polygon,
            bbox = shifted_bbox,
            polygon = new_polygon
        )
    

class FlorenceRegionLineODDataset(BaseImgXMLDataset):
    """Dataset that returns a region and bounding boxes of lines in the region"""

    def __init__(self, data_dir: str | Path):
        super().__init__(data_dir)
        self.task               = FlorenceTask.OD
        self.box_quantizer      = BoxQuantizer("floor", (1000, 1000))
        self.coords_quantizer   = CoordinatesQuantizer("floor", (1000, 1000))

        # Validate that the xml file has the data type we need, then set self.img_paths and self.xml_paths
        self.img_paths = []
        self.xml_paths = []
        self.img_paths, self.xml_paths = self.validate_and_load(self._all_img_paths, self._all_xml_paths)

    def validate_and_load(self, img_paths, xml_paths):
        valid_img_paths = []
        valid_xml_paths = []
        self.region_to_img_path = []

        # Pre-load region data from all XMLs
        self.regions_data = []
        for img, xml in zip(img_paths, xml_paths):
            regions = self.xmlparser.get_regions(xml)   # Fields: region_id, bbox, polygon, transcription

            if len(regions) > 0:
                self.regions_data += regions    # List of individual regions, not grouped by page
                self.region_to_img_path += [img] * len(regions)

                valid_img_paths.append(img)
                valid_xml_paths.append(xml)
        
        return valid_img_paths, valid_xml_paths

    @property
    def nsamples(self):
        return len(self.regions_data)
    
    def _get_one(self, idx):
        image       = Image.open(self.region_to_img_path[idx]).convert("RGB")
        data        = self.regions_data[idx]
        unique_key  = data["unique_key"]

        # Crop region image
        bbox_coords = bbox_xyxy_to_polygon(data["bbox"])
        region_img  = crop_image(image, bbox_coords)
        
        # Shift bbox and polygon to follow the newly cropped images
        shift_x = data["bbox"][0]
        shift_y = data["bbox"][1]

        shifted_bboxes = []
        bbox_texts = []
        for line in data["lines"]:
            shifted_bbox = (
                line["bbox"][0] - shift_x, 
                line["bbox"][1] - shift_y,
                line["bbox"][2] - shift_x, 
                line["bbox"][3] - shift_y
            )
            shifted_bboxes.append(shifted_bbox)

            # Convert bbox and polygon to florence text format
            florence_bbox = bbox_xyxy_to_florence(shifted_bbox, self.box_quantizer, region_img)
            bbox_texts.append("line" + florence_bbox)

        answer = "".join(bbox_texts)

        # question         = self.task,
        #     answer           = answer,
        #     image            = image,
        #     original_bboxes  = bboxes,
        #     quantized_bboxes = quantized_bboxes,
        #     image_path       = self.img_paths[idx],
        #     xml_path         = self.xml_paths[idx]

        # Form input question
        return dict(
            question        = self.task,
            answer          = answer,
            image           = region_img,
            original_bboxes = shifted_bboxes,
            unique_key      = unique_key,
        )
    

class FlorencePageTextODDataset(BaseImgXMLDataset):

    def __init__(
        self, 
        data_dir: str | Path,
        object_class: str = "region"
    ):
        assert object_class in ["region", "line"]
        super().__init__(data_dir=data_dir)
        
        self.object_class = object_class
        self.task = FlorenceTask.OD
        self.user_prompt = None
        self.box_quantizer = BoxQuantizer(mode="floor", bins=(1000, 1000))

        # Validate that the xml file has the data type we need, then set self.img_paths and self.xml_paths
        self.img_paths = []
        self.xml_paths = []
        self.img_paths, self.xml_paths = self.validate_and_load(self._all_img_paths, self._all_xml_paths)

    @property
    def nsamples(self):
        return len(self.img_paths)

    def validate_and_load(self, img_paths, xml_paths):
        valid_img_paths = []
        valid_xml_paths = []
        self.lines_data = []
        self.region_data = []

        for img, xml in zip(img_paths, xml_paths):
            lines = self.xmlparser.get_lines(xml)       # Fields: region_id, line_id, bbox, polygon, transcription
            regions = self.xmlparser.get_regions(xml)   # Fields: region_id, bbox, polygon, transcription

            if len(lines) > 0 and self.object_class == "line":
                valid_img_paths.append(img)
                valid_xml_paths.append(xml)
                self.lines_data += lines
            
            elif len(regions) > 0 and self.object_class == "region":
                valid_img_paths.append(img)
                valid_xml_paths.append(xml)
                self.region_data += regions
        
        return valid_img_paths, valid_xml_paths

    def _get_one(self, idx):
        image = Image.open(self.img_paths[idx]).convert("RGB")
        xml = self.xml_paths[idx]
        
        if self.object_class == "region":
            objects = self.xmlparser.get_regions(xml)
        elif self.object_class == "line":
            objects = self.xmlparser.get_lines(xml)  

        # Original bbox in xyxy format
        # Quantize bbox to coordinates relative to 1000 bins
        bboxes              = [data["bbox"] for data in objects]
        quantized_bboxes    = self.box_quantizer.quantize(torch.Tensor(bboxes), size=image.size)

        # # Convert bbox info to text
        bbox_texts = []
        for bbox in quantized_bboxes:
            bbox_text = self.object_class + "".join([f"<loc_{val}>" for val in bbox])
            bbox_texts.append(bbox_text)

        # Output text is of format "object_class<loc_...><loc_...><loc_...><loc_...>..."
        # bbox Format: xyxy
        answer = "".join(bbox_texts)
        
        return dict(
            question         = self.task,
            answer           = answer,
            image            = image,
            original_bboxes  = bboxes,
            image_path       = self.img_paths[idx],
            xml_path         = self.xml_paths[idx],
            unique_key       = self.img_paths[idx].stem
        )
    

# From https://huggingface.co/microsoft/Florence-2-large-ft/blob/main/processing_florence2.py
class BoxQuantizer(object):
    def __init__(self, mode, bins):
        self.mode = mode
        self.bins = bins

    def quantize(self, boxes: torch.Tensor, size):
        bins_w, bins_h = self.bins  # Quantization bins.
        size_w, size_h = size       # Original image size.
        size_per_bin_w = size_w / bins_w
        size_per_bin_h = size_h / bins_h
        xmin, ymin, xmax, ymax = boxes.split(1, dim=-1)  # Shape: 4 * [N, 1].

        if self.mode == 'floor':
            quantized_xmin = (
                xmin / size_per_bin_w).floor().clamp(0, bins_w - 1)
            quantized_ymin = (
                ymin / size_per_bin_h).floor().clamp(0, bins_h - 1)
            quantized_xmax = (
                xmax / size_per_bin_w).floor().clamp(0, bins_w - 1)
            quantized_ymax = (
                ymax / size_per_bin_h).floor().clamp(0, bins_h - 1)

        elif self.mode == 'round':
            raise NotImplementedError()

        else:
            raise ValueError('Incorrect quantization type.')

        quantized_boxes = torch.cat(
            (quantized_xmin, quantized_ymin, quantized_xmax, quantized_ymax), dim=-1
        ).int()

        return quantized_boxes

    def dequantize(self, boxes: torch.Tensor, size):
        bins_w, bins_h = self.bins  # Quantization bins.
        size_w, size_h = size       # Original image size.
        size_per_bin_w = size_w / bins_w
        size_per_bin_h = size_h / bins_h
        xmin, ymin, xmax, ymax = boxes.split(1, dim=-1)  # Shape: 4 * [N, 1].

        if self.mode == 'floor':
            # Add 0.5 to use the center position of the bin as the coordinate.
            dequantized_xmin = (xmin + 0.5) * size_per_bin_w
            dequantized_ymin = (ymin + 0.5) * size_per_bin_h
            dequantized_xmax = (xmax + 0.5) * size_per_bin_w
            dequantized_ymax = (ymax + 0.5) * size_per_bin_h

        elif self.mode == 'round':
            raise NotImplementedError()

        else:
            raise ValueError('Incorrect quantization type.')

        dequantized_boxes = torch.cat(
            (dequantized_xmin, dequantized_ymin,
             dequantized_xmax, dequantized_ymax), dim=-1
        )

        return dequantized_boxes    


class CoordinatesQuantizer(object):
    """
    Quantize coornidates (Nx2)
    """

    def __init__(self, mode, bins):
        self.mode = mode
        self.bins = bins

    def quantize(self, coordinates: torch.Tensor, size):
        bins_w, bins_h = self.bins  # Quantization bins.
        size_w, size_h = size       # Original image size.
        size_per_bin_w = size_w / bins_w
        size_per_bin_h = size_h / bins_h
        assert coordinates.shape[-1] == 2, 'coordinates should be shape (N, 2)'
        x, y = coordinates.split(1, dim=-1)  # Shape: 4 * [N, 1].

        if self.mode == 'floor':
            quantized_x = (x / size_per_bin_w).floor().clamp(0, bins_w - 1)
            quantized_y = (y / size_per_bin_h).floor().clamp(0, bins_h - 1)

        elif self.mode == 'round':
            raise NotImplementedError()

        else:
            raise ValueError('Incorrect quantization type.')

        quantized_coordinates = torch.cat(
            (quantized_x, quantized_y), dim=-1
        ).int()

        return quantized_coordinates

    def dequantize(self, coordinates: torch.Tensor, size):
        bins_w, bins_h = self.bins  # Quantization bins.
        size_w, size_h = size       # Original image size.
        size_per_bin_w = size_w / bins_w
        size_per_bin_h = size_h / bins_h
        assert coordinates.shape[-1] == 2, 'coordinates should be shape (N, 2)'
        x, y = coordinates.split(1, dim=-1)  # Shape: 4 * [N, 1].

        if self.mode == 'floor':
            # Add 0.5 to use the center position of the bin as the coordinate.
            dequantized_x = (x + 0.5) * size_per_bin_w
            dequantized_y = (y + 0.5) * size_per_bin_h

        elif self.mode == 'round':
            raise NotImplementedError()

        else:
            raise ValueError('Incorrect quantization type.')

        dequantized_coordinates = torch.cat(
            (dequantized_x, dequantized_y), dim=-1
        )

        return dequantized_coordinates



def create_collate_fn(processor, device):
    def func(batch):
        questions = [data["question"] for data in batch]
        answers = [data["answer"] for data in batch]
        images = [data["image"] for data in batch]
        
        inputs = processor(text=questions, images=images, return_tensors="pt", padding=True).to(device)
        labels = processor.tokenizer(
            text=answers, 
            return_tensors="pt", 
            padding=True, 
            return_token_type_ids=False
        ).input_ids.to(device)
        
        return dict(
            input_ids=inputs["input_ids"], 
            pixel_values=inputs["pixel_values"], 
            labels=labels,
        )

    return func


def predict(
    model, 
    processor, 
    images: Sequence[PILImage], 
    task_prompt: FlorenceTask = None, 
    user_prompt: str = None, 
    device: str = "cpu", 
) -> tuple[list, list]:
    
    # if task involve regions, need to concat task name and user prompt
    if task_prompt is not None:
        if user_prompt is None: 
            input_text = task_prompt
        else:
            input_text = task_prompt + user_prompt
    else:
        input_text = user_prompt

    inputs = processor(text=[input_text] * len(images), images=images, return_tensors="pt").to(device)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
        num_beams=3,
    )

    raw_output = processor.batch_decode(generated_ids, skip_special_tokens=False)
    parsed_output = []

    if task_prompt is not None:
        for idx, raw in enumerate(raw_output):
            parsed = processor.post_process_generation(raw, task=task_prompt, image_size=images[idx].size)
            parsed_output.append(parsed)
    
    return raw_output, parsed_output