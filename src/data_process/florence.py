import sys
import typing
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from pagexml.model.pagexml_document_model import PageXMLTextLine, PageXMLTextRegion, PageXMLPage

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from src.data_process.xml import XMLParser


class RunningTextDataset(Dataset):

    def __init__(self, data, question: str = "<SwedishHTR>"):
        self.data = data
        self.question = question

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        question = self.question
        answer = example["transcription"]
        image = example['image'].convert("RGB")
        return dict(
            question=question, 
            answer=answer, 
            image=image
        )
    
    def select(self, indices: typing.Iterable):
        subset = [self.data[int(idx)] for idx in indices]
        return RunningTextDataset(subset)


def construct_bbox(xml_obj: PageXMLTextLine | PageXMLTextRegion):

    seg_x = [x for (x, y) in xml_obj.coords.points]
    seg_y = [y for (x, y) in xml_obj.coords.points]

    xmin = min(seg_x)
    ymin = min(seg_y)
    xmax = max(seg_x)
    ymax = max(seg_y)

    # Can return this if want to draw using Rectangle
    anchor_x = xmin
    anchor_y =  ymin
    width = xmax - xmin
    height = ymax - ymin

    return xmin, ymin, xmax, ymax
    

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


class FlorenceTask():
    OD = "<OD>"
    DENSE_REGION_CAPTION = "<DENSE_REGION_CAPTION>"
    CAPTION = "<CAPTION>"
    DETAILED_CAPTION = "<DETAILED_CAPTION> "
    MORE_DETAILED_CAPTION = "<MORE_DETAILED_CAPTION> "
    REGION_PROPOSAL = "<REGION_PROPOSAL>"
    CAPTION_TO_PHRASE_GROUNDING = "<CAPTION_TO_PHRASE_GROUNDING>"
    REFERRING_EXPRESSION_SEGMENTATION = "<REFERRING_EXPRESSION_SEGMENTATION>"
    REGION_TO_SEGMENTATION = "<REGION_TO_SEGMENTATION>"
    OPEN_VOCABULARY_DETECTION = "<OPEN_VOCABULARY_DETECTION>"
    REGION_TO_DESCRIPTION = "<REGION_TO_DESCRIPTION>"
    OCR = "<OCR>"
    OCR_WITH_REGION = "<OCR_WITH_REGION>"

    def __init__(self):
        pass


class FlorenceTextODDataset(Dataset):

    def __init__(self, imgs_xmls: list[tuple | list], object_class: str = "region"):
        assert object_class in ["region", "line"]
        super().__init__()
        
        self.object_class = object_class
        self.task = FlorenceTask.OD
        self.user_prompt = None
        self.box_quantizer = BoxQuantizer(mode="floor", bins=(1000, 1000))
        self.xmlparser = XMLParser()

        # Validate that the xml files have regions
        valid_pairs = []
        objects = []
        for img, xml in imgs_xmls:
            if self.object_class == "region":
                objects = self.xmlparser.get_regions(xml)
            elif self.object_class == "line":
                objects = self.xmlparser.get_lines(xml)

            if len(objects) > 0:
                valid_pairs.append((img, xml))
        
        self.imgs_xmls = valid_pairs

    def __len__(self):
        return len(self.imgs_xmls)
    
    def __getitem__(self, idx):
        image = Image.open(self.imgs_xmls[idx][0]).convert("RGB")
        xml = self.imgs_xmls[idx][1]
        
        if self.object_class == "region":
            objects = self.xmlparser.get_regions(xml)
        elif self.object_class == "line":
            objects = self.xmlparser.get_lines(xml)

        bboxes = [data["bbox"] for data in objects]

        # Quantize bbox to coordinates relative to 1000 bins
        quantized_bboxes = self.box_quantizer.quantize(torch.Tensor(bboxes), size=image.size)

        # Convert bbox info to text
        bbox_texts = []
        for bbox in quantized_bboxes:
            bbox_text = self.object_class + "".join([f"<loc_{val}>" for val in bbox])
            bbox_texts.append(bbox_text)

        # Output text is of format </s><s>object_class<loc_...><loc_...><loc_...><loc_...>...</s>
        answer = "</s><s>" + "".join(bbox_texts) + "</s>"
        
        return dict(
            question=self.task,
            answer=answer,
            image=image,
            bboxes=bboxes
        )

    def select(self, indices: typing.Iterable):
        pairs = [self.imgs_xmls[idx] for idx in indices]
        return FlorenceTextODDataset(pairs)
