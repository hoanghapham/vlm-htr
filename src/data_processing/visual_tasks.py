from shapely import Polygon
from typing import Self, Iterable
from PIL import Image
from utils import XMLParser
from pathlib import Path


def bbox_xyxy_to_polygon(bbox):
    x1, y1, width, height = bbox_xyxy_to_xywh(bbox)

    # Order polygon points counter-clockwise
    x2 = x1 + height
    y2 = y1

    x3 = x1 + width
    y3 = y1 + height

    x4 = x1 + width
    y4 = y1

    return Polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])


def bbox_xyxy_to_xywh(bbox):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    return x1, y1, w, h


def bbox_xywh_to_xyxy(bbox):
    x1, y1, width, height = bbox
    x2 = x1 + width
    y2 = y1 + height
    return x1, y1, x2, y2


class Bbox():
    def __init__(self, xyxy: tuple):
        self.xyxy = xyxy
        self.xywh = bbox_xyxy_to_xywh(xyxy)
        self.polygon = bbox_xyxy_to_polygon(xyxy)

    @staticmethod
    def from_xywh(xywh: tuple) -> Self:
        xyxy = bbox_xywh_to_xyxy(xywh)
        return Bbox(xyxy)

    @staticmethod
    def from_polygon_coords(coords: list[tuple]) -> Self:
        x_coords = [tup[0] for tup in coords]
        y_coords = [tup[1] for tup in coords]
        x1 = min(x_coords)
        y1 = min(y_coords)
        x2 = max(x_coords)
        y2 = max(y_coords)
        return Bbox(xyxy=(x1, y1, x2, y2))
    
    @property
    def area(self):
        return self.polygon.area
    
    def __getitem__(self, idx):
        return self.xyxy[idx]

    def __repr__(self):
        return str(self.xyxy)


class TextRegionDataset():

    def __init__(self, img_paths: list[str | Path], xml_paths: list[str | Path]):
        matched = set([path.stem for path in img_paths]).intersection(set([path.stem for path in xml_paths]))
        assert len(img_paths) == len(xml_paths) == len(matched) > 0, \
            f"Length invalid, or mismatch img-xml pairs: {len(img_paths)} images, {len(xml_paths)} XML files, {len(matched)} matches"

        # Validate that the xml files have regions
        self.img_paths = []
        self.xml_paths = []
        self.xmlparser = XMLParser()

        objects = []
        for img, xml in zip(img_paths, xml_paths):
            objects = self.xmlparser.get_regions(xml)

            if len(objects) > 0:
                self.img_paths.append(img)
                self.xml_paths.append(xml)

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert("RGB")
        xml = self.xml_paths[idx]
        objects = self.xmlparser.get_regions(xml)
        bboxes = [Bbox(data["bbox"]) for data in objects]
        return dict(
            image=image,
            bboxes=bboxes
        )

    def select(self, indices: Iterable):
        img_paths = [self.img_paths[idx] for idx in indices]
        xml_paths = [self.xml_paths[idx] for idx in indices]
        return TextRegionDataset(img_paths, xml_paths)
    