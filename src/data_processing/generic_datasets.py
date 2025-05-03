
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from abc import ABC, abstractmethod
from typing import Iterable
from PIL import Image

from src.data_processing.visual_tasks import IMAGE_EXTENSIONS, crop_image
from src.file_tools import list_files
from src.data_processing.utils import XMLParser


class BaseImgXMLDataset(ABC):
    """Base dataset that load data from folders containing images & XML pairs
    The folder structure should be something like this:
    parent_folder/
        child_folder/
            images/
                image1.jpg
                image2.jpg
            page_xmls/
                image1.xml
                image2.xml
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        img_paths = list_files(self.data_dir, IMAGE_EXTENSIONS)
        xml_paths = list_files(self.data_dir, [".xml"])
        
        # Validate that the img and xml files match
        matched = set([path.stem for path in img_paths]).intersection(set([path.stem for path in xml_paths]))

        assert len(img_paths) == len(xml_paths) == len(matched) > 0, \
            f"Length invalid, or mismatch img-xml pairs: {len(img_paths)} images, {len(xml_paths)} XML files, {len(matched)} matches"

        # Validate that the xml files have regions or lines
        self.img_paths = []
        self.xml_paths = []
        self.xmlparser = XMLParser()

        self.validate_data(img_paths, xml_paths)

    def __len__(self):
        return len(self.img_paths)
    
    def select(self, indices: Iterable):
        for idx in indices:
            yield self.__getitem__(idx)
    
    @abstractmethod
    def __getitem__(self, idx):
        pass

    def validate_data(self, img_paths, xml_paths):
        for img, xml in zip(img_paths, xml_paths):
            assert img.stem == xml.stem, "File names mismatch"
            lines = self.xmlparser.get_lines(xml)
            regions = self.xmlparser.get_regions(xml)

            if len(lines) > 0 and len(regions) > 0:
                self.img_paths.append(img)
                self.xml_paths.append(xml)
    
    
class PageRegionODDataset(BaseImgXMLDataset):

    def __init__(self, data_dir: str | Path):
        super().__init__(data_dir=data_dir)

    def __getitem__(self, idx):
        img_filename = Path(self.img_paths[idx]).stem
        img_volume = Path(self.img_paths[idx]).parent.name
        image = Image.open(self.img_paths[idx]).convert("RGB")
        xml = self.xml_paths[idx]
        objects = self.xmlparser.get_regions(xml)
        bboxes = [data["bbox"] for data in objects]
        return dict(
            image=image,
            bboxes=bboxes,
            img_volume=img_volume,
            img_filename=img_filename,
            img_path=self.img_paths[idx],
            xml_path=self.xml_paths[idx]
        )


class RegionLineODDataset(BaseImgXMLDataset):

    def __init__(self, data_dir: str | Path):
        super().__init__(data_dir=data_dir)

        self.idx_to_img_path: list[Path] = []
        self.idx_to_xml_path: list[Path] = []
        self.region_bboxes = []
        self.region_polygons = []
        self.region_lines = []

        # Preload region - line data
        for idx, (img, xml) in enumerate(zip(self.img_paths, self.xml_paths)):
            regions = self.xmlparser.get_regions(xml)
            
            for region in regions:
                if len(region["lines"]) > 0:
                    self.idx_to_img_path.append(img)
                    self.idx_to_xml_path.append(xml)
                    self.region_bboxes.append(region["bbox"])
                    self.region_polygons.append(region["polygon"])
                    self.region_lines.append(region["lines"])


    def __getitem__(self, idx):
        """Return one region image & line bboxes"""
        img_filename    = self.idx_to_img_path[idx].stem
        img_volume      = self.idx_to_img_path[idx].parent.name
        full_image      = Image.open(self.idx_to_img_path[idx]).convert("RGB")

        region_polygon = self.region_polygons[idx]
        region_image = crop_image(full_image, region_polygon)

        region_lines = self.region_lines[idx]
        line_bboxes = [data["bbox"] for data in region_lines]

        return dict(
            image=region_image,
            bboxes=line_bboxes,
            img_volume=img_volume,
            img_filename=img_filename,
            img_path=self.idx_to_img_path[idx],
            xml_path=self.idx_to_img_path[idx]
        )


class PageLineODDataset(BaseImgXMLDataset):

    def __init__(self, data_dir: str | Path):
        super().__init__(data_dir=data_dir)

    def __getitem__(self, idx):
        img_filename = Path(self.img_paths[idx]).stem
        img_volume = Path(self.img_paths[idx]).parent.name
        image = Image.open(self.img_paths[idx]).convert("RGB")
        xml = self.xml_paths[idx]
        objects = self.xmlparser.get_lines(xml)
        bboxes = [data["bbox"] for data in objects]
        return dict(
            image=image,
            bboxes=bboxes,
            img_volume=img_volume,
            img_filename=img_filename,
            img_path=self.img_paths[idx],
            xml_path=self.xml_paths[idx]
        )

