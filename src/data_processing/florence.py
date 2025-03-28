import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

import typing
import torch
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pagexml.model.pagexml_document_model import PageXMLTextLine, PageXMLTextRegion, PageXMLPage

from src.data_processing.utils import XMLParser, load_arrow_datasets
from src.visualization import random_color
from src.file_tools import list_files


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
    
    def __str__(self):
        pass


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
        return FlorenceOCRDataset(subset)


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


class FlorenceTextODDataset(Dataset):

    def __init__(
        self, 
        data_dir: str | Path,
        task: FlorenceTask = FlorenceTask.OD, 
        object_class: str = "region", 
    ):
        self.data_dir = Path(data_dir)
        img_paths = list_files(self.data_dir / "images", [".tif", ".jpg"])
        xml_paths = list_files(self.data_dir / "page_xmls", [".xml"])
        matched = set([path.stem for path in img_paths]).intersection(set([path.stem for path in xml_paths]))
        assert len(img_paths) == len(xml_paths) == matched > 0, \
            f"Length invalid, or mismatch img-xml pairs: {len(img_paths)} images, {len(xml_paths)} XML files"
        
        assert object_class in ["region", "line"]
        super().__init__()
        
        self.object_class = object_class
        self.task = task
        self.user_prompt = None
        self.box_quantizer = BoxQuantizer(mode="floor", bins=(1000, 1000))
        self.xmlparser = XMLParser()

        # Validate that the xml files have regions
        self.img_paths = []
        self.xml_paths = []

        objects = []
        for img, xml in zip(img_paths, xml_paths):
            if self.object_class == "region":
                objects = self.xmlparser.get_regions(xml)
            elif self.object_class == "line":
                objects = self.xmlparser.get_lines(xml)

            if len(objects) > 0:
                self.img_paths.append(img)
                self.xml_paths.append(xml)

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert("RGB")
        xml = self.xml_paths[idx]
        
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

        # Output text is of format "object_class<loc_...><loc_...><loc_...><loc_...>..."
        answer = "".join(bbox_texts)
        
        return dict(
            question=self.task,
            answer=answer,
            image=image,
        )

    def select(self, indices: typing.Iterable):
        return [self.__getitem__(idx) for idx in indices]


def draw_bbox(image, data):
    """
    Plot BBox
    """
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(image)

    for bbox, label in zip(data['bboxes'], data['labels']):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1),
                                 x2 - x1,
                                 y2 - y1,
                                 linewidth=2,
                                 edgecolor='lime',
                                 facecolor='none')
        ax.add_patch(rect)
        plt.text(x1,
                 y1,
                 label,
                 color='black',
                 fontsize=8,
                 bbox=dict(facecolor='lime', alpha=1))

    ax.axis('off')
    plt.show()


# def draw_ocr_bboxes(image, prediction):
#     """
#     Draw OCR BBox
#     """
#     scale = 1
#     draw = ImageDraw.Draw(image)
#     bboxes, labels = prediction['quad_boxes'], prediction['labels']

#     for box, label in zip(bboxes, labels):
#         color = 'lime'
#         new_box = (np.array(box) * scale).tolist()
#         draw.polygon(new_box, width=4, outline=color)
#         draw.text((new_box[0] + 8, new_box[1] + 2),
#                   "{}".format(label),
#                   align="right",
#                   fill=color)
    
#     display(image)



def draw_seg_mask(img_obj: Image, bboxes: list, polygons: list, size: int=10):

    colors = [random_color() for _ in range(len(bboxes))]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(size, size))

    # Show the image
    ax.imshow(img_obj)

    # Draw the bounding box
    for idx, ann in enumerate(polygons):
        bbox = bboxes[idx]
        segm = polygons[idx]
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2], bbox[3],
            linewidth=2, edgecolor='lime', facecolor='none', label="Bounding Box"
        )
        ax.add_patch(rect)
        ax.text(bbox[0], bbox[1] - 10, str(idx), color = "lime", size=11)

        # Draw the segmentation mask
        seg_x = segm[0::2]
        seg_y = segm[1::2]
        ax.fill(seg_x, seg_y, facecolor=colors[idx], alpha=0.4, edgecolor=colors[idx], linewidth=2, label="Segmentation")

    # Set axis limits
    ax.set_xlim(0, img_obj.width)
    ax.set_ylim(img_obj.height, 0)  # Invert y-axis to match image coordinates

    # Labels and legend
    # ax.set_title(file_name)
    # ax.legend()
    # Show the plot
    plt.show()


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
    user_prompt: str, 
    image: Image, 
    task: FlorenceTask = None, 
    device: str = "cpu", 
) -> tuple[list, list]:
    
    if task:
        input_text = task
    else:
        input_text = user_prompt

    inputs = processor(text=input_text, images=image, return_tensors="pt").to(device)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
        num_beams=3,
    )

    raw_output = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_output = None

    if task:
        parsed_output = processor.post_process_generation(raw_output, task=task, image_size=image.size)
    
    return raw_output, parsed_output