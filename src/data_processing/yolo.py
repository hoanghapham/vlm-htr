
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from PIL import Image

from shapely.geometry import Polygon
from src.data_processing.visual_tasks import crop_image, bbox_xyxy_to_polygon
from src.data_processing.base_datasets import BaseImgXMLDataset


# Conversion between normal format and YOLO

def bbox_xyxy_to_yolo_format(bbox: tuple | list, img_width: int, img_height: int, class_id=0) -> str:
    """Convert xyxy bbox to yolo string

    Parameters
    ----------
    bbox : tuple | list
    img_width : int
    img_height : int
    class_id : int, optional
        Class of the instance, by default 0

    Returns
    -------
    str
        Format: {class_id} {x_center} {y_center} {width} {height}
    """
    assert len(bbox) == 4, f"bbox has {len(bbox)} elements"
    (xmin, ymin, xmax, ymax) = bbox
    x_center = (xmin + xmax) / 2 / img_width
    y_center = (ymin + ymax) / 2 / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    return f"{class_id} {x_center} {y_center} {width} {height}"


def bboxes_xyxy_to_yolo_format(bboxes: list[tuple], img_width: int, img_height: int, class_id=0) -> list[str]:
    """Accept a list of bboxes in xyxy format and convert to YOLO format:
        {class_id} {x_center} {y_center} {width} {height}

    Parameters
    ----------
    bboxes : list[tuple]
    img_width : int
    img_height : int
    class_id : int, optional
        Class of the object, by default 0

    Returns
    -------
    list[str]
        List of YOLO formatted annotations
    """
    yolo_annotations = []
    for bbox in bboxes:
        yolo_str = bbox_xyxy_to_yolo_format(bbox, img_width, img_height, class_id=class_id)
        yolo_annotations.append(yolo_str)
    return yolo_annotations


def polygon_to_yolo_format(polygon: list[list | tuple] | Polygon, image_width: int, image_height: int, class_id=0) -> str:
    """Convert polygon coordinates to YOLO instance segmentation format.
        

    Parameters
    ----------
    coords : list[list  |  tuple]
    image_width : int
    image_height : int
    class_id : int, optional
        Class of the instance, by default 0

    Returns
    -------
    str
        str: YOLO segmentation formatted annotation string.
    """
    normalized_points = []

    if isinstance(polygon, Polygon):
        coords = polygon.exterior.coords
    else:
        coords = polygon

    for x, y in coords:
        normalized_x = x / image_width
        normalized_y = y / image_height
        normalized_points.extend([normalized_x, normalized_y])
    
    annotation = f"{class_id} " + " ".join(f"{p:.6f}" for p in normalized_points)
    return annotation


def yolo_seg_to_polygon(yolo_annotation: str, image_width: int, image_height: int) -> tuple[int, Polygon]:
    """
    Convert YOLO instance segmentation format back to polygon coordinates.
    
    Args:
        yolo_annotation (str): YOLO formatted annotation string.
        image_width (int): Width of the image.
        image_height (int): Height of the image.
    
    Returns:
        list: List of (x, y) coordinates in original scale.
    """
    parts = yolo_annotation.split()
    class_id = int(parts[0])
    normalized_points = list(map(float, parts[1:]))
    
    coords = []
    for i in range(0, len(normalized_points), 2):
        x = normalized_points[i] * image_width
        y = normalized_points[i + 1] * image_height
        coords.append([x, y])
    
    polygon = Polygon(coords)
    return class_id, polygon

    
class YOLOPageRegionODDataset(BaseImgXMLDataset):

    def __init__(self, data_dir: str | Path):
        super().__init__(data_dir=data_dir)

        # Validate that the xml file has the data type we need, then set self.img_paths and self.xml_paths
        self.img_paths = []
        self.xml_paths = []
        self.img_paths, self.xml_paths = self.validate_and_load(self._all_img_paths, self._all_xml_paths)

    @property
    def nsamples(self):
        return len(self.img_paths)

    def validate_and_load(self, img_paths, xml_paths):
        # List to convert a global line idx to path of an image
        valid_img_paths = []
        valid_xml_paths = []

        self.regions_to_img_path = []
        
        # Pre-load regions data from all XMLs
        self.page_regions_data = []

        for idx, xml in enumerate(xml_paths):
            regions = self.xmlparser.get_regions(xml)  # Fields: region_id, line_id, bbox, polygon, transcription
            
            if len(regions) > 0:
                valid_img_paths.append(img_paths[idx])
                valid_xml_paths.append(xml)

            self.page_regions_data.append(regions)
        
        return valid_img_paths, valid_xml_paths

    def _get_one(self, idx):
        img_filename    = Path(self.img_paths[idx]).stem
        img_volume      = Path(self.img_paths[idx]).parent.name
        image           = Image.open(self.img_paths[idx]).convert("RGB")
        bboxes          = [data["bbox"] for data in self.page_regions_data[idx]]
        yolo_bboxes     = bboxes_xyxy_to_yolo_format(bboxes, image.width, image.height)

        return dict(
            image=image,
            bboxes=bboxes,
            yolo_bboxes=yolo_bboxes,
            img_volume=img_volume,
            img_filename=img_filename,
            unique_key=img_filename,
            img_path=self.img_paths[idx],
            xml_path=self.xml_paths[idx]
        )


class YOLORegionLineODDataset(BaseImgXMLDataset):

    def __init__(self, data_dir: str | Path):
        super().__init__(data_dir=data_dir)

        # Validate that the xml file has the data type we need, then set self.img_paths and self.xml_paths
        self.img_paths = []
        self.xml_paths = []
        self.img_paths, self.xml_paths = self.validate_and_load(self._all_img_paths, self._all_xml_paths)

    @property
    def nsamples(self):
        return len(self.region_ids)

    def validate_and_load(self, img_paths, xml_paths):
        valid_img_paths = []
        valid_xml_paths = []
        self.idx_to_img_path: list[Path] = []
        self.idx_to_xml_path: list[Path] = []
        self.region_ids = []
        self.region_bboxes = []
        self.region_polygons = []
        self.region_lines_raw = []
        self.region_lines_shifted = []

        # Preload region - line data
        for img, xml in zip(img_paths, xml_paths):
            regions = self.xmlparser.get_regions(xml)
            
            for region in regions:
                if len(region["lines"]) > 0:
                    valid_img_paths.append(img)
                    valid_xml_paths.append(xml)

                    self.idx_to_img_path.append(img)
                    self.idx_to_xml_path.append(xml)
                    self.region_ids.append(region["region_id"])
                    self.region_bboxes.append(region["bbox"])
                    self.region_polygons.append(region["polygon"])
                    self.region_lines_raw.append(region["lines"])

                    # Shift lines to match region bbox
                    shifted_bboxes = []
                    shift_x = region["bbox"][0]
                    shift_y = region["bbox"][1]

                    for line in region["lines"]:
                        shifted = (
                            line["bbox"][0] - shift_x, 
                            line["bbox"][1] - shift_y,
                            line["bbox"][2] - shift_x, 
                            line["bbox"][3] - shift_y
                        )
                        shifted_bboxes.append(shifted)

                    self.region_lines_shifted.append(shifted_bboxes)

        return valid_img_paths, valid_xml_paths

    def _get_one(self, idx):
        """Return one region image & line bboxes"""
        img_filename    = self.idx_to_img_path[idx].stem
        img_volume      = self.idx_to_img_path[idx].parent.name
        full_image      = Image.open(self.idx_to_img_path[idx]).convert("RGB")

        region_polygon  = self.region_polygons[idx]
        region_image    = crop_image(full_image, region_polygon)
        bboxes          = self.region_lines_shifted[idx]
        yolo_bboxes     = bboxes_xyxy_to_yolo_format(bboxes, region_image.width, region_image.height)

        return dict(
            image=region_image,
            bboxes=bboxes,
            yolo_bboxes=yolo_bboxes,
            img_volume=img_volume,
            img_filename=img_filename,
            unique_key=f"{img_filename}_{self.region_ids[idx]}",
            img_path=self.idx_to_img_path[idx],
            xml_path=self.idx_to_img_path[idx]
        )


class YOLOPageLineODDataset(BaseImgXMLDataset):

    def __init__(self, data_dir: str | Path):
        super().__init__(data_dir=data_dir)

        # Validate that the xml file has the data type we need, then set self.img_paths and self.xml_paths
        self.img_paths = []
        self.xml_paths = []
        self.img_paths, self.xml_paths = self.validate_and_load(self._all_img_paths, self._all_xml_paths)

    @property
    def nsamples(self):
        return len(self.img_paths)
    
    def validate_and_load(self, img_paths, xml_paths):
        # List to convert a global line idx to path of an image
        valid_img_paths = []
        valid_xml_paths = []

        self.line_to_img_path = []
        
        # Pre-load lines data from all XMLs
        self.page_lines_data = []

        for idx, xml in enumerate(xml_paths):
            lines = self.xmlparser.get_lines(xml)  # Fields: region_id, line_id, bbox, polygon, transcription
            
            if len(lines) > 0:
                valid_img_paths.append(img_paths[idx])
                valid_xml_paths.append(xml)

            self.page_lines_data.append(lines)
        
        return valid_img_paths, valid_xml_paths

    def _get_one(self, idx):
        img_filename    = Path(self.img_paths[idx]).stem
        img_volume      = Path(self.img_paths[idx]).parent.name
        image           = Image.open(self.img_paths[idx]).convert("RGB")
        bboxes          = [data["bbox"] for data in self.page_lines_data[idx]]
        yolo_bboxes     = bboxes_xyxy_to_yolo_format(bboxes, image.width, image.height)
        
        return dict(
            unique_key=img_filename,
            image=image,
            bboxes=bboxes,
            yolo_bboxes=yolo_bboxes,
            img_volume=img_volume,
            img_filename=img_filename,
            img_path=self.img_paths[idx],
            xml_path=self.xml_paths[idx]
        )


class YOLORegionLineSegDataset():
    """Line instance segmentation dataset for YOLO is created using this script:
    pipelines/data_process/create_yolo_data/create_inst_seg_lines_within_regions.py
    """
    def __init__(self, data_dir: str | Path):
        pass


class YOLOSingleLineSegDataset(BaseImgXMLDataset):
    """Dataset that returns one rectangular crop of a line, with polygon seg mask"""

    def __init__(self, data_dir: str | Path):
        super().__init__(data_dir)

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
            
            if lines > 0:
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
        bbox_polygoin = bbox_xyxy_to_polygon(data["bbox"])
        line_img = crop_image(image, bbox_polygoin)
        
        # Shift bbox and polygon to follow the newly cropped images
        shift_x = data["bbox"][0]
        shift_y = data["bbox"][1]

        new_bbox = (
            0, 
            0,
            data["bbox"][2] - shift_x, 
            data["bbox"][3] - shift_y
        )

        new_polygon = Polygon([(x - shift_x, y - shift_y) for (x, y) in data["polygon"]])
        yolo_polygon = polygon_to_yolo_format(new_polygon, line_img.width, line_img.height)

        # Convert bbox and polygon to florence text format

        return dict(
            unique_key = unique_key,
            image = line_img,
            bbox = new_bbox,
            polygon = new_polygon,
            yolo_polygon = yolo_polygon,
        )