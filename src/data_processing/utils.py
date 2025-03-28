import xml.etree.ElementTree as ET
from pathlib import Path
from torch.utils.data import Dataset
from datasets import load_from_disk, concatenate_datasets

class XMLParser():
    def __init__(self, verbose: bool = False):
        self.verbose = verbose    
        
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

    def get_regions(self, xml_path: str | Path):
        """Parses the PAGE XML and extracts region data."""
        root = self._parse_xml(xml_path)
        if not root:
            return []

        namespaces = {"ns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
        regions_data = []
        for region in root.findall(".//ns:TextRegion", namespaces):
            if region is not None:
                try:
                    region_id = region.get("id")
                    polygon = self._get_polygon(region, namespaces)
                    bbox = self._get_bbox(polygon)
                    transcription = region.find("ns:TextEquiv/ns:Unicode", namespaces).text or ""
                    regions_data.append({"region_id": region_id, "bbox": bbox, "polygon": polygon, "transcription": transcription})
                except Exception as e:
                    if self.verbose:
                        print(f"Error parsing region: {e}")
        return regions_data
        
    def get_lines(self, xml_path):
        """Parses the PAGE XML and extracts line data."""
        root = self._parse_xml(xml_path)
        if not root:
            return []

        namespaces = {"ns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
        lines_data = []
        for region in root.findall(".//ns:TextRegion", namespaces):
            for line in region.findall(".//ns:TextLine", namespaces):
                if line is not None:
                    try:
                        region_id = line.get("id")
                        polygon = self._get_polygon(line, namespaces)
                        bbox = self._get_bbox(polygon)
                        transcription = line.find("ns:TextEquiv/ns:Unicode", namespaces).text or ""
                        lines_data.append({"region_id": region_id, "bbox": bbox, "polygon": polygon, "transcription": transcription})
                    except Exception as e:
                        if self.verbose:
                            print(f"Error parsing line: {e}")
        return lines_data
    

def load_arrow_datasets(parent_dir: str | Path) -> Dataset:
    dsets = []
    dir_paths = [path for path in parent_dir.glob("*") if path.is_dir()]
    for path in dir_paths:
        try:
            data = load_from_disk(path)
            dsets.append(data)
        except Exception as e:
            print(e)

    dataset = concatenate_datasets(dsets)
    return dataset
