from shapely.geometry import Polygon
from htrflow.utils.geometry import Bbox
from typing import Self
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


class ODOutput():
    def __init__(self, bboxes: list[Bbox], polygons: list[Polygon]):
        assert len(bboxes) == len(polygons), "Number of bboxes and polygons must be equal"

        self.bboxes = bboxes
        self.polygons = polygons

    def __len__(self):
        return len(self.bboxes)
    
    def _get_one(self, idx):
        return dict(bbox=self.bboxes[idx], polygon=self.polygons[idx])
    
    def __add__(self, other: Self):
        return ODOutput(bboxes=self.bboxes + other.bboxes, polygons=self.polygons + other.polygons)
        
    def __getitem__(self, idx: int | slice):
        if isinstance(idx, int):
            return self._get_one(idx)
        else:
            return [self._get_one(i) for i in range(idx.start or 0, idx.stop or -1, idx.step or 1) if i < len(self.bboxes)]
        
