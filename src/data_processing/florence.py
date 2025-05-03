import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from typing import Iterable, Sequence
import torch
from torch.utils.data import Dataset
from PIL import Image

from src.data_processing.utils import XMLParser, load_arrow_datasets
from src.data_processing.visual_tasks import BaseImgXMLDataset, polygon_to_bbox_xyxy, crop_image, bbox_xyxy_to_coords


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


def bboxes_xyxy_to_florence(bboxes, box_quantizer, image):
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


# class FlorenceRegionLineODDataset(Dataset):
#     """Receive a cached arrow dataset created from the HTRDatasetBuilder.line_od_within_regions() method.
#     Input data are region image, output are bbox of lines
#     """
#     def __init__(self, data: Dataset, task: FlorenceTask = FlorenceTask.REGION_TO_SEGMENTATION):
#         self.data = data
#         self.task = task
#         self.box_quantizer      = BoxQuantizer(mode="floor", bins=(1000, 1000))
#         self.coords_quantizer   = CoordinatesQuantizer(mode="floor", bins=(1000, 1000))

#         total_lines = 0
#         reg_to_lines = {}
#         line_to_reg = []
#         line_to_local_line = []

#         # Create map
#         current_idx = 0

#         for reg_idx, sample in enumerate(data):
#             n_lines = len(sample["annotations"])
#             total_lines += n_lines
#             reg_to_lines[reg_idx] = list(range(current_idx, current_idx + n_lines))
            
#             # List containing duplicates of reg_idx to match (global) line idx to region
#             line_to_reg += [reg_idx] * n_lines

#             # List containing local line idx, used to map global to local linie idx
#             line_to_local_line += list(range(n_lines))

#             current_idx += n_lines
        
#         self.total_lines = total_lines
#         self.reg_to_lines = reg_to_lines
#         self.line_to_reg = line_to_reg
#         self.line_to_local_line = line_to_local_line

#     def __len__(self):
#         return self.total_lines

#     def __getitem__(self, idx):
#         """Return one line by translating the global dataset's line idx to region's local line idx"""
        
#         reg_idx = self.line_to_reg[idx]                 # Find the region idx from the global line idx
#         local_line_idx = self.line_to_local_line[idx]   # Convert global line idx to a local line idx

#         example = self.data[reg_idx]                # Get one region
#         image = example["image"].convert("RGB")     

#         line    = example["annotations"][local_line_idx]   # Get one line within the region
#         bbox = line["bbox"]
#         bboxes  = coords_to_bbox_xyxy(polygon)

#         florence_bbox       = bboxes_xyxy_to_florence([bboxes], self.box_quantizer, image)[0]
#         florence_polygon    = polygons_to_florence([polygon], self.coords_quantizer, image)[0]
        
#         question = self.task + florence_bbox

#         return dict(
#             task=self.task,
#             question=question,
#             answer=florence_polygon,
#             image=image 
#         )
    
#     def select(self, indices: Iterable):
#         return [self.__getitem__(idx) for idx in indices]


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

        # List to convert a global line idx to path of an image
        self.line_to_img_path = []

        # Pre-load lines data from all XMLs
        # Fields: region_id, line_id, bbox, polygon, transcription
        self.lines_data = []
        for idx, xml in enumerate(self.xml_paths):
            lines = self.xmlparser.get_lines(xml)
            self.lines_data += lines
            self.line_to_img_path += [self.img_paths[idx]] * len(lines)

    def __len__(self):
        return len(self.lines_data)

    def _get_one(self, idx):
        image       = Image.open(self.line_to_img_path[idx]).convert("RGB")
        data        = self.lines_data[idx]
        unique_key  = data["unique_key"]

        # Crop image to a line using bbox
        bbox_coords = bbox_xyxy_to_coords(data["bbox"])
        cropped_line_img = crop_image(image, bbox_coords)
        
        # Shift bbox and polygon to follow the newly cropped images
        shift_x = data["bbox"][0]
        shift_y = data["bbox"][1]

        new_bbox = (
            0, 
            0,
            data["bbox"][2] - shift_x, 
            data["bbox"][3] - shift_y
        )

        new_polygon = [(x - shift_x, y - shift_y) for (x, y) in data["polygon"]]

        # Convert bbox and polygon to florence text format
        florence_bbox   = bbox_xyxy_to_florence(new_bbox, self.box_quantizer, cropped_line_img)
        florence_polygon = coords_to_florence(new_polygon, self.coords_quantizer, cropped_line_img)

        # Form input question
        question = self.task + florence_bbox

        return dict(
            unique_key = unique_key,
            image = cropped_line_img,
            question = question,
            answer = florence_polygon,
            bbox = new_bbox,
            polygon = new_polygon
        )
    
    def __getitem__(self, idx: int | slice):
        if isinstance(idx, int):
            return self._get_one(idx)
        else:
            return [self._get_one(i) for i in range(idx.start, idx.stop, idx.step or 1) if i < len(self.lines_data)]


class FlorenceRegionLineODDataset(BaseImgXMLDataset):
    """Dataset that returns a region and bounding boxes of lines in the region"""

    def __init__(self, data_dir: str | Path):
        super().__init__(data_dir)
        self.task               = FlorenceTask.OD
        self.box_quantizer      = BoxQuantizer("floor", (1000, 1000))
        self.coords_quantizer   = CoordinatesQuantizer("floor", (1000, 1000))

        # List to convert a global line idx to path of an image
        self.region_to_img_path = []

        # Pre-load region data from all XMLs
        # Fields: region_id, bbox, polygon, transcription
        # TODO: Define a function to get region data, and get lines data from region
        self.regions_data = []
        for idx, xml in enumerate(self.xml_paths):
            regions = self.xmlparser.get_regions(xml)
            self.regions_data += regions
            self.region_to_img_path += [self.img_paths[idx]] * len(regions)

    def __len__(self):
        return len(self.regions_data)
    

    def _get_one(self, idx):
        image       = Image.open(self.region_to_img_path[idx]).convert("RGB")
        data        = self.regions_data[idx]
        unique_key  = data["unique_key"]

        # Crop region image
        bbox_coords         = bbox_xyxy_to_coords(data["bbox"])
        cropped_region_img  = crop_image(image, bbox_coords)
        
        # Shift bbox and polygon to follow the newly cropped images
        shift_x = data["bbox"][0]
        shift_y = data["bbox"][1]

        texts = []
        for line in data["lines"]:
            new_bbox = (
                line["bbox"][0] - shift_x, 
                line["bbox"][1] - shift_y,
                line["bbox"][2] - shift_x, 
                line["bbox"][3] - shift_y
            )

            # Convert bbox and polygon to florence text format
            florence_bbox   = bbox_xyxy_to_florence(new_bbox, self.box_quantizer, cropped_region_img)
            texts.append("line" + florence_bbox)

        answer = "".join(texts)


        # Form input question

        return dict(
            unique_key = unique_key,
            image = cropped_region_img,
            question = FlorenceTask.OD,
            answer = answer,
        )
    
    def __getitem__(self, idx: int | slice):
        if isinstance(idx, int):
            return self._get_one(idx)
        else:
            return [self._get_one(i) for i in range(idx.start, idx.stop, idx.step or 1) if i < len(self.regions_data)]


class FlorencePageTextODDataset(BaseImgXMLDataset):

    def __init__(
        self, 
        data_dir: str | Path,
        task: FlorenceTask = FlorenceTask.OD, 
        object_class: str = "region", 
    ):
        assert object_class in ["region", "line"]
        super().__init__(data_dir=data_dir)
        
        self.object_class = object_class
        self.task = task
        self.user_prompt = None
        self.box_quantizer = BoxQuantizer(mode="floor", bins=(1000, 1000))

    def _get_one(self, idx):
        image = Image.open(self.img_paths[idx]).convert("RGB")
        xml = self.xml_paths[idx]
        
        if self.object_class == "region":
            objects = self.xmlparser.get_regions(xml)
        elif self.object_class == "line":
            objects = self.xmlparser.get_lines(xml)  

        # Original bbox in xyxy format
        bboxes = [data["bbox"] for data in objects]

        # Quantize bbox to coordinates relative to 1000 bins
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
            question=self.task,
            answer=answer,
            image=image,
            original_bboxes=bboxes,
            quantized_bboxes=quantized_bboxes,
            image_path=self.img_paths[idx],
            xml_path=self.xml_paths[idx]
        )
    
    def __getitem__(self, idx: int | slice):
        if isinstance(idx, int):
            return self._get_one(idx)
        else:
            return [self._get_one(i) for i in range(idx.start, idx.stop, idx.step or 1) if i < len(self.img_paths)]


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
    images: Sequence[Image], 
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