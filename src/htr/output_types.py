from shapely.geometry import Polygon
from htrflow.utils.geometry import Bbox

class Line():
    def __init__(self, bbox: Bbox = None, polygon: Polygon = None, text: str = None):
        self.bbox = bbox
        self.polygon = polygon
        self.text = text


class Region():
    def __init__(self, bbox: Bbox = None, polygon: Polygon = None, lines: list[Line] = None):
        self.bbox = bbox
        self.polygon = polygon
        self.lines = lines
        self.text = " ".join([line.text for line in lines])


class Page():
    def __init__(self, regions: list[Region] = None, lines: list[Line] = None):
        self.regions = regions
        self.lines = lines
        self.text = " ".join([line.text for line in lines])