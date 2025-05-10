import io, os, sys
from pathlib import Path, PurePath
from glob import glob
sys.path.append(str(Path(__file__).parent.parent.parent))

import cv2
import numpy as np
from datasets import (
    BuilderConfig,
    DatasetInfo,
    Features,
    GeneratorBasedBuilder,
    Image,
    Sequence,
    Split,
    SplitGenerator,
    Value,
)
from PIL import Image as PILImage
from htrflow.utils.geometry import Bbox
from shapely.geometry import Polygon
import xml.etree.ElementTree as ET


IMAGE_EXTENSIONS = [
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".tif",
            ".tiff",
            ".JPG",
            ".JPEG",
            ".PNG",
            ".GIF",
            ".BMP",
            ".TIF",
            ".TIFF",]


def bbox_xyxy_to_polygon(bbox: list[tuple]) -> Polygon:
    x1, y1, width, height = bbox_xyxy_to_xywh(bbox)

    # Order polygon points counter-clockwise
    x2 = x1 
    y2 = y1 + height

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


def polygon_to_bbox_xyxy(polygon: Polygon | list[tuple[int, int]]):

    if isinstance(polygon, Polygon):
        boundary = polygon.boundary.coords
    else:
        boundary = polygon

    x_coords = [tup[0] for tup in boundary]
    y_coords = [tup[1] for tup in boundary]

    x1 = min(x_coords)
    y1 = min(y_coords)
    x2 = max(x_coords)
    y2 = max(y_coords)
    return x1, y1, x2, y2


def get_cover_bbox(bboxes: list[Bbox]) -> Bbox:
    """Compute the minimum bounding box that covers all input boxes.

    Args:
        boxes: List of bounding boxes (xmin, ymin, xmax, ymax)

    Returns:
        A tuple (xmin, ymin, xmax, ymax) of the minimal covering box.
    """
    if not bboxes:
        raise ValueError("Input list of boxes is empty.")

    xmin = min(box[0] for box in bboxes)
    ymin = min(box[1] for box in bboxes)
    xmax = max(box[2] for box in bboxes)
    ymax = max(box[3] for box in bboxes)

    return Bbox(xmin, ymin, xmax, ymax)


def sort_polygons(polygons, y_threshold=10):
    """
    Sort polygons top-to-bottom, left-to-right within horizontal bands.
    
    Args:
        polygons: list of list of (x, y) tuples
        y_threshold: vertical tolerance to consider two masks on the same line
        
    Returns:
        Sorted list of polygons
    """
    # Extract min y and min x for each polygon

    def polygon_key(poly):
        ys = [pt[1] for pt in poly]
        xs = [pt[0] for pt in poly]
        return (min(ys), min(xs))

    # Sort polygons initially by min y and then min x
    try:
        polygons_sorted = sorted(polygons, key=polygon_key)
    except Exception:
        return []

    # Now group by horizontal line (based on y threshold)
    grouped = []
    current_group = [polygons_sorted[0]]
    current_y = min(pt[1] for pt in polygons_sorted[0])

    for poly in polygons_sorted[1:]:
        min_y = min(pt[1] for pt in poly)
        if abs(min_y - current_y) <= y_threshold:
            current_group.append(poly)
        else:
            grouped.append(sorted(current_group, key=lambda p: min(pt[0] for pt in p)))
            current_group = [poly]
            current_y = min_y

    grouped.append(sorted(current_group, key=lambda p: min(pt[0] for pt in p)))

    # Flatten the grouped list
    return [poly for group in grouped for poly in group]


def crop_image(img, polygon: Polygon | list[tuple[int, int]]):
    """Crops an image based on the provided polygon coordinates. 
    Apply a white background for areas outside of the polygon"""
    image_array = np.array(img)
    # point_list = [(point[0], point[1]) for point in polygon]

    if isinstance(polygon, list):
        polygon = Polygon(polygon)

    points_array = np.array(list(polygon.boundary.coords)).astype(int)

    mask = np.zeros(image_array.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [points_array], -1, (255, 255, 255), -1, cv2.LINE_AA)


    # Apply mask to image
    res = cv2.bitwise_and(image_array, image_array, mask=mask)
    rect = cv2.boundingRect(points_array)

    # Create a white background and overlay the cropped image
    wbg = np.ones_like(image_array, np.uint8) * 255
    cv2.bitwise_not(wbg, wbg, mask=mask)
    dst = wbg + res

    cropped = dst[rect[1] : rect[1] + rect[3], rect[0] : rect[0] + rect[2]]

    # cv2_image_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    return PILImage.fromarray(cropped).convert("RGB")


# Reuse code from Riksarkivet, with some modifications
# https://huggingface.co/datasets/Riksarkivet/goteborgs_poliskammare_fore_1900/blob/main/goteborgs_poliskammare_fore_1900.py
class HTRDatasetConfig(BuilderConfig):
    """Configuration for each dataset variant."""

    def __init__(self, name, description, process_func, features, **kwargs):
        super().__init__(name=name, description=description, **kwargs)
        self.process_func = process_func
        self.features = features


class HTRDatasetBuilder(GeneratorBasedBuilder):
    # Define feature structures for each dataset type
    text_recognition_features = Features(
        {
            "image": Image(),
            "transcription": Value("string"),
        }
    )

    segmentation_features = Features(
        {
            "image_name": Value("string"),
            "image": Image(),
            "annotations": Sequence(
                {
                    "polygon": Sequence(Sequence(Value("float32"))),
                    "transcription": Value("string"),
                    "class": Value("string"),
                }
            ),
        }
    )

    BUILDER_CONFIGS = [
        HTRDatasetConfig(
            name="text_recognition__line_seg",
            description="textline dataset for text recognition of historical Swedish",
            process_func="text_recognition__line_seg",
            features=text_recognition_features,
        ),
        HTRDatasetConfig(
            name="text_recognition__line_bbox",
            description="textline dataset for text recognition within bounding box",
            process_func="text_recognition__line_bbox",
            features=text_recognition_features,
        ),
        HTRDatasetConfig(
            name="inst_seg_lines_within_regions",
            description="Cropped text region images with text line annotations",
            process_func="inst_seg_lines_within_regions",
            features=segmentation_features,
        ),
        HTRDatasetConfig(
            name="line_od_within_regions",
            description="Cropped text region images with line bbox annotations",
            process_func="inst_seg_lines_within_regions",
            features=segmentation_features,
        ),
        HTRDatasetConfig(
            name="inst_seg_regions_and_lines",
            description="Original images with both region and line annotations",
            process_func="inst_seg_regions_and_lines",
            features=segmentation_features,
        ),
        HTRDatasetConfig(
            name="inst_seg_lines",
            description="Original images with text line annotations only",
            process_func="inst_seg_lines",
            features=segmentation_features,
        ),
        HTRDatasetConfig(
            name="inst_seg_regions",
            description="Original images with text region annotations only",
            process_func="inst_seg_regions",
            features=segmentation_features,
        ),
    ]

    def _info(self):
        return DatasetInfo(features=self.config.features)

    def _split_generators(self, dl_manager):
        # Define URLs for images and XMLs
        """
        images_url = [
            f"https://huggingface.co/datasets/Riksarkivet/ra_enstaka_sidor/resolve/main/data/images/ra_enstaka_sidor_images_{i}.tar.gz"
            for i in range(1, 3)
        ]
        xmls_url = [
            f"https://huggingface.co/datasets/Riksarkivet/ra_enstaka_sidor/resolve/main/data/page_xmls/ra_enstaka_sidor_page_xmls_{i}.tar.gz"
            for i in range(1, 3)
        ]

        """

        images = dl_manager.download_and_extract(
                            [
                                f"https://huggingface.co/datasets/Riksarkivet/gota_hovratt_seg/resolve/main/data/images/gota_hovratt_seg_images_{i}.tar.gz"
                                for i in range(1, 3)
                            ]
                        )
        xmls = dl_manager.download_and_extract(
                            [
                                f"https://huggingface.co/datasets/Riksarkivet/gota_hovratt_seg/resolve/main/data/page_xmls/gota_hovratt_seg_page_xmls_{i}.tar.gz"
                                for i in range(1, 3)
                            ]
                        )

        # Download and extract images and XMLs
        # images = dl_manager.download_and_extract(images_url)
        # xmls = dl_manager.download_and_extract(xmls_url)

        # Define supported image file extensions
        image_extensions = [
            "*.jpg",
            "*.jpeg",
            "*.png",
            "*.gif",
            "*.bmp",
            "*.tif",
            "*.tiff",
            "*.JPG",
            "*.JPEG",
            "*.PNG",
            "*.GIF",
            "*.BMP",
            "*.TIF",
            "*.TIFF",
        ]

        # Collect and sort image and XML file paths
        imgs_flat = self._collect_file_paths(images, image_extensions)
        xmls_flat = self._collect_file_paths(xmls, ["*.xml"])

        # Ensure the number of images matches the number of XML files
        assert len(imgs_flat) == len(xmls_flat)

        # Pair images and XML files
        imgs_xmls = list(
            zip(sorted(imgs_flat, key=lambda x: Path(x).stem), sorted(xmls_flat, key=lambda x: Path(x).stem))
        )

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={"imgs_xmls": imgs_xmls},
            )
        ]

    def _collect_file_paths(self, folders, extensions):
        """Collects file paths recursively from specified folders."""
        files_nested = [
            glob(os.path.join(folder, "**", ext), recursive=True) for ext in extensions for folder in folders
        ]
        return [file for sublist in files_nested for file in sublist]

    def _generate_examples(self, imgs_xmls):
        process_func = getattr(self, self.config.process_func)
        return process_func(imgs_xmls)
    
    def text_recognition__line_seg(self, imgs_xmls):
        """Process for line dataset with cropped images and transcriptions."""
        for img, xml in imgs_xmls:
            img_filename, volume = self._extract_filename_and_volume(img, xml)
            lines_data = self.parse_pagexml(xml)
            image_array = cv2.imread(img)

            for i, line in enumerate(lines_data):
                line_id = str(i).zfill(4)
                try:
                    cropped_image = self.crop_line_image(image_array, line["coords"])
                except Exception as e:
                    print(f"Error image: {img_filename}: {e}")
                    continue
                
                transcription = line["transcription"]

                if not transcription:
                    print(f"Invalid transcription: {transcription}")
                    continue

                unique_key = f"{volume}_{img_filename}_{line_id}"
                yield {
                    "unique_key": unique_key, 
                    "img_filename": img_filename, 
                    "image": cropped_image, 
                    "transcription": transcription
                }

    def text_recognition__line_bbox(self, imgs_xmls):
        """Process for line dataset with cropped images from bbox, and transcriptions."""
        for img, xml in imgs_xmls:
            img_filename, volume = self._extract_filename_and_volume(img, xml)
            lines_data = self.parse_pagexml(xml)
            image_array = cv2.imread(img)

            for i, line in enumerate(lines_data):
                line_id = str(i).zfill(4)
                bbox        = polygon_to_bbox_xyxy(line["coords"])
                bbox_coords = bbox_xyxy_to_polygon(bbox).boundary.coords
                try:
                    cropped_image = self.crop_line_image(image_array, bbox_coords)
                except Exception as e:
                    print(f"Error image: {img_filename}: {e}")
                    continue 

                transcription = line["transcription"]

                if not transcription:
                    print(f"Invalid transcription: {transcription}")
                    continue

                unique_key = f"{volume}_{img_filename}_{line_id}"
                yield {
                    "unique_key": unique_key, 
                    "img_filename": img_filename, 
                    "image": cropped_image, 
                    "transcription": transcription
                }

    def inst_seg_lines_within_regions(self, imgs_xmls):
        """Process for cropped images with text line annotations."""
        for img_path, xml_path in imgs_xmls:
            img_filename, volume = self._extract_filename_and_volume(img_path, xml_path)
            image = PILImage.open(img_path)
            root = self._parse_xml(xml_path)
            namespaces = {"ns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}

            # Iterate through each TextRegion
            for reg_ind, region in enumerate(root.findall(".//ns:TextRegion", namespaces=namespaces)):
                reg_id = str(reg_ind).zfill(4)
                region_polygon = self._get_polygon(region, namespaces)
                min_x, min_y, max_x, max_y = self._get_bbox(region_polygon)
                cropped_region_image = self.crop_image(image, region_polygon)

                annotations = self._get_line_annotations_within_region(
                    region, namespaces, min_x, min_y, region_polygon
                )

                unique_key = f"{volume}_{img_filename}_{reg_id}"
                try:
                    yield {
                            "unique_key": unique_key,
                            "img_filename": img_filename,
                            # "image": {"bytes": self._image_to_bytes(cropped_region_image)},
                            "image": cropped_region_image,
                            "annotations": annotations,
                        }
                except Exception as e:
                    print("still error", e)
                    continue
    

    def line_od_within_regions(self, imgs_xmls):
        """Process for cropped region images with line bbox annotations."""
        for img_path, xml_path in imgs_xmls:
            img_filename, volume = self._extract_filename_and_volume(img_path, xml_path)
            image = PILImage.open(img_path)
            root = self._parse_xml(xml_path)
            namespaces = {"ns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}

            # Iterate through each TextRegion
            for reg_ind, region in enumerate(root.findall(".//ns:TextRegion", namespaces=namespaces)):
                reg_id = str(reg_ind).zfill(4)
                region_polygon = self._get_polygon(region, namespaces)
                min_x, min_y, max_x, max_y = self._get_bbox(region_polygon)
                cropped_region_image = self.crop_image(image, region_polygon)

                annotations = self._get_line_bbox_within_region(
                    region, namespaces, min_x, min_y, region_polygon
                )

                unique_key = f"{volume}_{img_filename}_{reg_id}"
                try:
                    yield {
                            "unique_key": unique_key,
                            "img_filename": img_filename,
                            "image": cropped_region_image,
                            "annotations": annotations,
                        }
                except Exception as e:
                    print("still error", e)
                    continue

    def inst_seg_regions_and_lines(self, imgs_xmls):
        """Process for original images with both region and line annotations."""
        for img_path, xml_path in imgs_xmls:
            img_filename, volume = self._extract_filename_and_volume(img_path, xml_path)
            image = PILImage.open(img_path)
            root = self._parse_xml(xml_path)
            annotations = self._get_region_and_line_annotations(root)

            unique_key = f"{volume}_{img_filename}"
            yield {
                "unique_key": unique_key, 
                "img_filename": img_filename, 
                "image": image, 
                "annotations": annotations
            }

    def inst_seg_lines(self, imgs_xmls):
        """Process for original images with text line annotations only."""
        for img_path, xml_path in imgs_xmls:
            img_filename, volume = self._extract_filename_and_volume(img_path, xml_path)
            image = PILImage.open(img_path)
            root = self._parse_xml(xml_path)

            annotations = self._get_line_annotations(root)

            unique_key = f"{volume}_{img_filename}"
            yield {
                "unique_key": unique_key, 
                "img_filename": img_filename, 
                "image": image, 
                "annotations": annotations
            }

    def inst_seg_regions(self, imgs_xmls):
        """Process for original images with text region annotations only."""
        for img_path, xml_path in imgs_xmls:
            img_filename, volume = self._extract_filename_and_volume(img_path, xml_path)
            image = PILImage.open(img_path)
            root = self._parse_xml(xml_path)

            annotations = self._get_region_annotations(root)

            unique_key = f"{volume}_{img_filename}"
            yield {
                "unique_key": unique_key, 
                "img_filename": img_filename, 
                "image": image, 
                "annotations": annotations
            }

    def _extract_filename_and_volume(self, img, xml):
        """Extracts the filename and volume from the image and XML paths."""
        assert Path(img).stem == Path(xml).stem
        img_filename = Path(img).stem
        volume = PurePath(img).parts[-2]
        return img_filename, volume

    def _parse_xml(self, xml_path):
        """Parses the XML file and returns the root element."""
        try:
            tree = ET.parse(xml_path)
            return tree.getroot()
        except ET.ParseError as e:
            print(f"XML Parse Error: {e}")
            return None

    def _get_line_annotations_within_region(self, region, namespaces, min_x, min_y, region_polygon):
        """Generates annotations for text lines within a region."""
        annotations = []
        for line in region.findall(".//ns:TextLine", namespaces=namespaces):
            line_polygon = self._get_polygon(line, namespaces)
            clipped_line_polygon = self.clip_polygon_to_region(line_polygon, region_polygon)

            if len(clipped_line_polygon) < 3:
                print(f"Invalid polygon detected for line: {line_polygon}, clipped: {clipped_line_polygon}")
                continue

            translated_polygon = [(x - min_x, y - min_y) for x, y in clipped_line_polygon]
            transcription = "".join(line.itertext()).strip()

            annotations.append(
                {
                    "polygon": translated_polygon,
                    "transcription": transcription,
                    "class": "textline",
                }
            )
        return annotations

    def _get_line_bbox_within_region(self, region, namespaces, min_x, min_y, region_polygon):
        """Generates annotations for text lines within a region."""
        annotations = []
        for line in region.findall(".//ns:TextLine", namespaces=namespaces):
            line_polygon = self._get_polygon(line, namespaces)
            clipped_line_polygon = self.clip_polygon_to_region(line_polygon, region_polygon)

            if len(clipped_line_polygon) < 3:
                print(f"Invalid polygon detected for line: {line_polygon}, clipped: {clipped_line_polygon}")
                continue

            translated_polygon = [(x - min_x, y - min_y) for x, y in clipped_line_polygon]
            bbox = polygon_to_bbox_xyxy(translated_polygon)
            # transcription = "".join(line.itertext()).strip()

            annotations.append(
                {
                    # "polygon": translated_polygon,
                    "bbox": bbox,
                    # "transcription": transcription,
                    "class": "line_bbox",
                }
            )
        return annotations

    def _get_region_and_line_annotations(self, root):
        """Generates annotations for both text regions and lines."""
        annotations = []

        # Get region annotations
        annotations.extend(self._get_region_annotations(root))

        # Get line annotations
        annotations.extend(self._get_line_annotations(root))

        return annotations

    def _get_line_annotations(self, root):
        """Generates annotations for text lines only."""
        namespaces = {"ns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
        annotations = []
        for region in root.findall(".//ns:TextRegion", namespaces=namespaces):
            for line in region.findall(".//ns:TextLine", namespaces=namespaces):
                line_polygon = self._get_polygon(line, namespaces)
                transcription = "".join(line.itertext()).strip()
                annotations.append(
                    {
                        "polygon": line_polygon,
                        "transcription": transcription,
                        "class": "textline",
                    }
                )
        return annotations

    def _get_region_annotations(self, root):
        """Generates annotations for text regions only."""
        namespaces = {"ns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
        annotations = []
        for region in root.findall(".//ns:TextRegion", namespaces=namespaces):
            region_polygon = self._get_polygon(region, namespaces)
            annotations.append(
                {
                    "polygon": region_polygon,
                    "transcription": "",
                    "class": "textregion",
                }
            )
        return annotations

    def _image_to_bytes(self, image):
        """Converts a PIL image to bytes."""
        with io.BytesIO() as output:
            image.save(output, format="PNG")
            return output.getvalue()

    def crop_image(self, img_pil, coords):
        coords = np.array(coords)
        img = np.array(img_pil)
        mask = np.zeros(img.shape[0:2], dtype=np.uint8)

        try:
            # Ensure the coordinates are within the bounds of the image
            coords[:, 0] = np.clip(coords[:, 0], 0, img.shape[1] - 1)
            coords[:, 1] = np.clip(coords[:, 1], 0, img.shape[0] - 1)

            # Draw the mask
            cv2.drawContours(mask, [coords], -1, (255, 255, 255), -1, cv2.LINE_AA)

            # Apply mask to image
            res = cv2.bitwise_and(img, img, mask=mask)
            rect = cv2.boundingRect(coords)

            # Ensure the bounding box is within the image dimensions
            rect = (
                max(0, rect[0]),
                max(0, rect[1]),
                min(rect[2], img.shape[1] - rect[0]),
                min(rect[3], img.shape[0] - rect[1]),
            )

            wbg = np.ones_like(img, np.uint8) * 255
            cv2.bitwise_not(wbg, wbg, mask=mask)

            # Overlap the resulted cropped image on the white background
            dst = wbg + res

            # Use validated rect for cropping
            cropped = dst[rect[1] : rect[1] + rect[3], rect[0] : rect[0] + rect[2]]

            # Convert the NumPy array back to a PIL image
            cropped_pil = PILImage.fromarray(cropped)

            return cropped_pil

        except Exception as e:
            print(f"Error in cropping: {e}")
            return img_pil  # Return the original image if there's an error

    def _create_mask(self, shape, coords):
        """Creates a mask for the specified polygon coordinates."""
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.drawContours(mask, [np.array(coords)], -1, (255, 255, 255), -1, cv2.LINE_AA)
        return mask

    def parse_pagexml(self, xml):
        """Parses the PAGE XML and extracts line data."""
        root = self._parse_xml(xml)
        if not root:
            return []

        namespaces = {"ns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
        lines_data = []
        for region in root.findall(".//ns:TextRegion", namespaces):
            for line in region.findall(".//ns:TextLine", namespaces):
                try:
                    line_id = line.get("id")
                    coords = self._get_polygon(line, namespaces)
                    transcription = line.find("ns:TextEquiv/ns:Unicode", namespaces).text or ""
                    lines_data.append({"line_id": line_id, "coords": coords, "transcription": transcription})
                except Exception as e:
                    print(f"Error parsing line: {e}")
        return lines_data

    def crop_line_image(self, img, coords):
        """Crops a line image based on the provided coordinates."""
        mask = self._create_mask(img.shape[:2], coords)

        coords = np.array(coords)

        # Apply mask to image
        res = cv2.bitwise_and(img, img, mask=mask)
        rect = cv2.boundingRect(coords)

        # Create a white background and overlay the cropped image
        wbg = np.ones_like(img, np.uint8) * 255
        cv2.bitwise_not(wbg, wbg, mask=mask)
        dst = wbg + res

        cropped = dst[rect[1] : rect[1] + rect[3], rect[0] : rect[0] + rect[2]]

        return self.cv2_to_pil(cropped)

    def _get_polygon(self, element, namespaces):
        """Extracts polygon points from a PAGE XML element."""
        coords = element.find(".//ns:Coords", namespaces=namespaces).attrib["points"]
        return [tuple(map(int, p.split(","))) for p in coords.split()]

    def _get_bbox(self, polygon):
        """Calculates the bounding box from polygon points."""
        min_x = min(p[0] for p in polygon)
        min_y = min(p[1] for p in polygon)
        max_x = max(p[0] for p in polygon)
        max_y = max(p[1] for p in polygon)
        return min_x, min_y, max_x, max_y

    def clip_polygon_to_region(self, line_polygon, region_polygon):
        """
        Clips a line polygon to ensure it's inside the region polygon using Shapely.
        Returns the original line polygon if the intersection is empty.
        """
        # Convert lists of points to Shapely Polygons
        line_poly = Polygon(line_polygon)
        region_poly = Polygon(region_polygon)

        # Compute the intersection of the line polygon with the region polygon
        try:
            intersection = line_poly.intersection(region_poly)
        except Exception:
            return line_polygon

        # Return the intersection points as a list of tuples
        if intersection.is_empty:
            print(
                f"No intersection found for line_polygon {line_polygon} within region_polygon {region_polygon}, returning original polygon."
            )
            return line_polygon
        elif intersection.geom_type == "Polygon":
            return list(intersection.exterior.coords)
        elif intersection.geom_type == "MultiPolygon":
            # If the result is a MultiPolygon, take the largest by area (or another heuristic)
            largest_polygon = max(intersection.geoms, key=lambda p: p.area)
            return list(largest_polygon.exterior.coords)
        elif intersection.geom_type == "LineString":
            return list(intersection.coords)
        else:
            print(f"Unexpected intersection type: {intersection.geom_type}")
            return line_polygon

    def cv2_to_pil(self, cv2_image):
        """Converts an OpenCV image to a PIL Image."""
        cv2_image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        return PILImage.fromarray(cv2_image_rgb)
    
