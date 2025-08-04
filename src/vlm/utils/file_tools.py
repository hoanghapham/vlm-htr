from pathlib import Path
import os
import sys
import json
import unicodedata
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from datasets import load_from_disk, concatenate_datasets


def list_files(input_path: Path | str, extensions) -> list[Path]:
    """Return a sorted list of PosixPaths"""
    if not isinstance(input_path, Path):
        input_path = Path(input_path)

    files = [
        file for file in sorted(list(input_path.glob("**/*"))) 
        if file.suffix in extensions
        and file.is_file()
    ]
        
    return files


class suppress_stdout_stderr(object):
    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')

        self.old_stdout_fileno_undup    = sys.stdout.fileno()
        self.old_stderr_fileno_undup    = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup ( sys.stdout.fileno() )
        self.old_stderr_fileno = os.dup ( sys.stderr.fileno() )

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2 ( self.outnull_file.fileno(), self.old_stdout_fileno_undup )
        os.dup2 ( self.errnull_file.fileno(), self.old_stderr_fileno_undup )

        sys.stdout = self.outnull_file        
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):        
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2 ( self.old_stdout_fileno, self.old_stdout_fileno_undup )
        os.dup2 ( self.old_stderr_fileno, self.old_stderr_fileno_undup )

        os.close ( self.old_stdout_fileno )
        os.close ( self.old_stderr_fileno )

        self.outnull_file.close()
        self.errnull_file.close()


def read_ndjson_file(input_path: Path | str) -> list[dict]:
    if not isinstance(input_path, Path):
        input_path = Path(input_path)

    data = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            data.append(json.loads(line))

    return data


def read_json_file(input_path: Path | str) -> dict:
    with open(input_path, "r") as f:
        data = json.load(f)
    return data


def write_ndjson_file(data: list[dict], output_path: Path | str):
    if not isinstance(output_path, Path):
        output_path = Path(output_path)
    
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")


def write_json_file(data: dict, output_path: Path | str):
    if not isinstance(output_path, Path):
        output_path = Path(output_path)

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")


def read_lines(input_file: Path | str) -> list[str]:
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    return lines


def read_text_file(input_file: Path | str) -> str:
    with open(input_file, "r", encoding="utf-8") as f:
        text = "".join(list(f.readlines()))
    return text


def write_text_file(text: str, output_path: Path | str) -> None:
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)


def write_list_to_text_file(lst: list[str], output_path: Path | str, linebreak=True) -> None:
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)
        
    with open(output_path, "w", encoding="utf-8") as f:
        for line in lst:
            f.write(line)
            if linebreak:
                f.write("\n")


def normalize_name(s):
    return unicodedata.normalize('NFD', s)



class XMLParser():
    """Class for parsing PAGE XML files."""
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
