import io
import xml.etree.ElementTree as ET
from pathlib import Path, PurePath

from pagexml.model.pagexml_document_model import PageXMLTextLine, PageXMLTextRegion, PageXMLPage
from pagexml.parser import parse_pagexml_file

import cv2
import numpy as np
from PIL import Image as PILImage
from shapely.geometry import Polygon
from shapely import MultiPoint, convex_hull
from tqdm import tqdm



def create_lower_bound(line_list: list[PageXMLTextLine]) -> list[tuple]:
    # lower bound points: select lower points of the last line. 
    lower_bound = []
    max_x = 0
    for point in line_list[-1].coords.points:
        if point[0] > max_x:
            lower_bound.append(point)
            max_x = point[0]
        else:
            break
    
    return lower_bound


def create_right_bound(line_list: list[PageXMLTextLine]) -> list[tuple]:
    # Right bound: For each line, select n rightmost point
    n_candidates = 2
    right_bound = []

    for line in reversed(line_list):
        # sort points by x to get the top n x-farthest points
        # Picking n candidates, sort them from bottom to top
        ordered_points = sorted(line.coords.points, key= lambda x: x[0], reverse=True)
        rightmost_candidates = sorted(ordered_points[:n_candidates], key = lambda x: x[1], reverse=True)
        right_bound += rightmost_candidates

    return right_bound


def create_upper_bound(line_list: list[PageXMLTextLine]) -> list[tuple]:
    # Upper bound: select upper points of the first line.
    
    upper_bound = []
    max_x = 0
    for point in reversed(line_list[0].coords.points):
        if point[0] > max_x:
            upper_bound.append(point)
            max_x = point[0]
    
    upper_bound = list(reversed(upper_bound))
    return upper_bound


def create_left_bound(line_list: list[PageXMLTextLine]) -> list[tuple]:
    # Left bound: for each line, select n leftmost point
    left_bound = []
    n_candidates = 2

    for line in line_list:
        ordered_points = sorted(line.coords.points, key= lambda x: x[0], reverse=False)
        leftmost_candidates = sorted(ordered_points[:n_candidates], key = lambda x: x[1], reverse=False)
        left_bound += leftmost_candidates
    
    return left_bound


def merge_polys(line_list: list[PageXMLTextLine]):
    # Need to dedup points as well

    lower_bound = create_lower_bound(line_list)
    right_bound = create_right_bound(line_list)
    upper_bound = create_upper_bound(line_list)
    left_bound = create_left_bound(line_list)
    
    # Remove outliners
    cleaned_right_bound = []
    rb_x = [point[0] for point in right_bound]
    rb_x_mean = np.median(rb_x)
    for point in right_bound:
        if point[0] < rb_x_mean - 1 * np.std(rb_x):
            continue
        cleaned_right_bound.append(point)

    # Remove outliers
    cleaned_left_bound = []
    lb_x = [point[0] for point in left_bound]
    lb_x_mean = np.median(lb_x)
    for point in left_bound:
        if point[0] > lb_x_mean + 1 * np.std(lb_x):
            continue
        cleaned_left_bound.append(point)

    # Construct region poly: points go in counter-clockwise , starting from lower bound
    region_poly = lower_bound + cleaned_right_bound + upper_bound + cleaned_left_bound
    
    return region_poly


# Group lines into regions within a char limit

def split_regions(xml_data: PageXMLPage, region_chars_limit=1024) -> list[list[PageXMLTextLine]]:

    regions = []
    for region_data in xml_data.get_all_text_regions():

        current_group = []
        current_length = 0

        # If region is within char limit, add region's line, then move to the next
        if region_data.text is not None and len(region_data.text) <= region_chars_limit:
            regions += [region_data.get_lines()]
        else:
            for line_idx, line_data in enumerate(region_data.get_lines()):

                if (line_idx == len(region_data.get_lines()) - 1):
                    current_group.append(line_data)
                    regions.append(current_group)

                if line_data.text is not None:

                    # If adding the current line will exceed the char limit, select the current line groups, then reset the list
                    if (current_length + len(line_data.text) > region_chars_limit):
                        regions.append(current_group)

                        # Reset
                        current_group = [line_data]
                        current_length = len(line_data.text)
                    
                    else:

                        # If not, continue to add line to the group
                        current_group.append(line_data)
                        current_length += len(line_data.text)
    return regions



def join_transcriptions(line_list: list[PageXMLTextLine]):
    texts = []
    for line in line_list:
        if line.text:
            texts.append(line.text)
    
    return "\n".join(texts)


class ImageDatasetBuilder():
    # Define feature structures for each dataset type

    def create_line_dataset(self, imgs_xmls):
        """Process for line dataset with cropped images and transcriptions."""
        for img, xml in tqdm(imgs_xmls, total=len(imgs_xmls), unit="page", desc="Processing"):
            img_filename, volume = self._extract_filename_and_volume(img, xml)
            lines_data = self.parse_pagexml(xml)
            image_array = cv2.imread(img)

            for i, line in enumerate(lines_data):
                region_id = str(i).zfill(4)
                try:
                    cropped_image = self.crop_line_image(image_array, line["coords"])
                except Exception as e:
                    print("Error image:", img_filename)
                    print(e)
                    yield None

                transcription = line["transcription"]

                if not transcription:
                    print(f"Invalid transcription: {transcription}")
                    yield None

                unique_key = f"{volume}_{img_filename}_{region_id}"
                yield {"unique_key": unique_key, "image": cropped_image, "transcription": transcription}


    def process_one_line(self, img, xml, line_idx):
        img_filename, volume = self._extract_filename_and_volume(img, xml)
        lines_data = self.parse_pagexml(xml)
        image_array = cv2.imread(img)
        region_id = str(line_idx).zfill(4)
        
        try:
            cropped_image = self.crop_line_image(image_array, lines_data[line_idx]["coords"])
        except Exception:
            print("Error image:", img_filename)
            return None

        transcription = lines_data[line_idx]["transcription"]

        if not transcription:
            print(f"Invalid transcription: {transcription}")
            return None

        unique_key = f"{volume}_{img_filename}_{region_id}"
        return {"unique_key": unique_key, "image": cropped_image, "transcription": transcription}



    def create_smooth_region_dataset(self, imgs_xmls):
        for img, xml in tqdm(imgs_xmls, total=len(imgs_xmls), unit="page", desc="Processing"):
            img_filename, volume = self._extract_filename_and_volume(img, xml)
            xml_data = parse_pagexml_file(xml)
            image_array = cv2.imread(img)

            regions = split_regions(xml_data)

            for i, region in enumerate(regions):
                
                # Create mask
                merged_lines = merge_polys(region)
                hull = convex_hull(MultiPoint(merged_lines))  # Find minimal convex hull that cover the merged lines
                mask = [(int(x), int(y)) for x, y in hull.boundary.coords]
                
                # Join transcription
                transcription = join_transcriptions(region)

                try:
                    cropped_image = self.crop_line_image(image_array, mask)
                except Exception as e:
                    print("Error image:", img_filename)
                    print(e)
                    continue

                region_id = str(i).zfill(4)
                unique_key = f"{volume}_{img_filename}_{region_id}"
                yield unique_key, {"image": cropped_image, "transcription": transcription}


    def create_wiggly_region_dataset(self, imgs_xmls):
        for img, xml in tqdm(imgs_xmls, total=len(imgs_xmls), unit="page", desc="Processing"):
            img_filename, volume = self._extract_filename_and_volume(img, xml)
            xml_data = parse_pagexml_file(xml)
            image_array = cv2.imread(img)

            regions = split_regions(xml_data)

            for i, region in enumerate(regions):

                mask = merge_polys(region)

                transcription = join_transcriptions(region)
                try:
                    cropped_image = self.crop_line_image(image_array, mask)
                except Exception as e:
                    print("Error image:", img_filename)
                    print(e)
                    continue

                region_id = str(i).zfill(4)
                unique_key = f"{volume}_{img_filename}_{region_id}"
                yield unique_key, {"image": cropped_image, "transcription": transcription}


    def create_poly_region_dataset(self, imgs_xmls):
        for img, xml in tqdm(imgs_xmls, total=len(imgs_xmls), unit="page", desc="Processing"):
            img_filename, volume = self._extract_filename_and_volume(img, xml)
            xml_data = parse_pagexml_file(xml)
            image_array = cv2.imread(img)

            regions = split_regions(xml_data)

            for i, region in enumerate(regions):

                mask = merge_polys(region)

                transcription = join_transcriptions(region)
                try:
                    cropped_image = self.crop_line_image(image_array, mask)
                except Exception as e:
                    print("Error image:", img_filename)
                    print(e)
                    continue


                region_id = str(i).zfill(4)
                unique_key = f"{volume}_{img_filename}_{region_id}"
                yield unique_key, {"image": cropped_image, "transcription": transcription}


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
                    region_id = line.get("id")
                    coords = self._get_polygon(line, namespaces)
                    transcription = line.find("ns:TextEquiv/ns:Unicode", namespaces).text or ""
                    lines_data.append({"region_id": region_id, "coords": coords, "transcription": transcription})
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
    

