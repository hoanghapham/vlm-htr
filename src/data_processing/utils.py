from pathlib import Path

import numpy as np
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from datasets import load_from_disk, concatenate_datasets


class XMLParser():
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.namespaces = {"ns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
        
    def _parse_xml(self, xml_path: str | Path):
        """Parses the XML file and returns the root element."""
        try:
            tree = ET.parse(xml_path)
            return tree.getroot()
        except ET.ParseError as e:
            if self.verbose:
                print(f"XML Parse Error: {e}")
            return None

    def _get_polygon(self, element, namespaces):
        """Extracts polygon points from a PAGE XML element."""
        polygon = element.find(".//ns:Coords", namespaces=namespaces).attrib["points"]
        return [tuple(map(int, p.split(","))) for p in polygon.split()]
    
    def _get_bbox(self, polygon):
        """Calculates the bounding box from polygon points."""
        min_x = min(p[0] for p in polygon)
        min_y = min(p[1] for p in polygon)
        max_x = max(p[0] for p in polygon)
        max_y = max(p[1] for p in polygon)
        return min_x, min_y, max_x, max_y
    
    def _extract_region_data(self, region):
        region_id = region.get("id")
        polygon = self._get_polygon(region, self.namespaces)
        bbox = self._get_bbox(polygon)
        transcription = region.find("ns:TextEquiv/ns:Unicode", self.namespaces).text or ""

        lines = []
        for line in region.findall(".//ns:TextLine", self.namespaces):
            if line is not None:
                try:
                    data = self._extract_line_data(region, line)
                    lines.append(data)
                except Exception as e:
                    if self.verbose:
                        print(f"Error parsing line: {e}")

        return {"region_id": region_id, "bbox": bbox, "polygon": polygon, "transcription": transcription, "lines": lines}

    def _extract_line_data(self, region, line):
        region_id = region.get("id")
        line_id = line.get("id")
        polygon = self._get_polygon(line, self.namespaces)
        bbox = self._get_bbox(polygon)
        transcription = line.find("ns:TextEquiv/ns:Unicode", self.namespaces).text or ""
        return {"region_id": region_id, "line_id": line_id, "bbox": bbox, "polygon": polygon, "transcription": transcription}

    def get_regions(self, xml_path: str | Path):
        """Parses the PAGE XML and extracts region data."""
        root = self._parse_xml(xml_path)
        if not root:
            return []
        img_filename = Path(xml_path).stem

        regions_data = []
        for region in root.findall(".//ns:TextRegion", self.namespaces):
            if region is not None:
                try:
                    data = self._extract_region_data(region)
                    regions_data.append(data)
                except Exception as e:
                    if self.verbose:
                        print(f"Error parsing region: {e}")
        
        for idx, data in enumerate(regions_data):
            idx_str = str(idx).zfill(4)
            data["unique_key"] = f"{img_filename}_{idx_str}"

        return regions_data
    
    def get_lines(self, xml_path):
        """Parses the PAGE XML and extracts line data."""
        root = self._parse_xml(xml_path)
        if not root:
            return []
        
        img_filename = Path(xml_path).stem
        
        lines_data = []
        for region in root.findall(".//ns:TextRegion", self.namespaces):
            for line in region.findall(".//ns:TextLine", self.namespaces):
                if line is not None:
                    try:
                        data = self._extract_line_data(region, line)
                        lines_data.append(data)
                    except Exception as e:
                        if self.verbose:
                            print(f"Error parsing line: {e}")
        
        for idx, data in enumerate(lines_data):
            idx_str = str(idx).zfill(4)
            data["unique_key"] = f"{img_filename}_{idx_str}"
        
        return lines_data
    
    def get_regions_with_lines(self, xml_path):
        root = self._parse_xml(xml_path)
        if not root:
            return []

        regions_data = []
        for region in root.findall(".//ns:TextRegion", self.namespaces):

            cur_region_data = {}

            # Find regions
            if region is not None:
                try:
                    cur_region_data = self._extract_region_data(region)
                except Exception as e:
                    if self.verbose:
                        print(f"Error parsing region: {e}")

                lines_data = []

                for line in region.findall(".//ns:TextLine", self.namespaces):
                    if line is not None:
                        try:
                            data = self._extract_line_data(region, line)
                            lines_data.append(data)
                        except Exception as e:
                            if self.verbose:
                                print(f"Error parsing line: {e}")
                
                cur_region_data["lines"] = lines_data
                regions_data.append(cur_region_data)

        return regions_data
    

def load_arrow_datasets(parent_dir: str | Path) -> Dataset:
    dsets = []
    dir_paths = [path for path in parent_dir.iterdir() if (path.is_symlink() or path.is_dir())]
    for path in dir_paths:
        try:
            data = load_from_disk(path)
            dsets.append(data)
        except Exception as e:
            print(e)

    dataset = concatenate_datasets(dsets)
    return dataset


def gen_split_indices(
    total_samples: int, 
    seed: int = 42, 
    train_ratio: float = 0.7, 
    val_ratio: float = 0.15, 
    test_ratio: float = 0.15
) -> tuple[list[int], list[int], list[int]]:
    np.random.seed(seed)
    all_indices = range(total_samples)

    train_indices = np.random.choice(all_indices, size=int(train_ratio * total_samples), replace=False)
    val_indices = np.random.choice(
        [idx for idx in all_indices if idx not in train_indices], 
        size = int(val_ratio * total_samples), 
        replace = False
    )
    test_indices = np.random.choice(
        [idx for idx in all_indices if idx not in np.concatenate([train_indices, val_indices])], 
        size = max(total_samples - len(train_indices) - len(val_indices), int(test_ratio * total_samples)),
        replace = False
    )

    return train_indices, val_indices, test_indices
