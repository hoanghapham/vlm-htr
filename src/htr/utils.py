import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

import numpy as np
from typing import Sequence
from PIL.Image import Image as PILImage
from htrflow.utils.geometry import Bbox
from shapely.geometry import Polygon
from htrflow.utils.layout import estimate_printspace, is_twopage as check_twopage, get_region_location
from src.evaluation.visual_metrics import compute_bbox_iou
from src.data_types import Page, Region, Line


# Code from https://github.com/AI-Riksarkivet/htrflow/blob/main/src/htrflow/postprocess/reading_order.py, with modifications

def sort_consider_margins(bboxes: Sequence[Bbox], image: PILImage) -> list[int]:
    """Order bounding boxes with respect to printspace, and consider margin.

    This function estimates the reading order based on the following:
        1. Which page of the spread the bounding box belongs to (if `is_twopage` is True)
        2. Where the bounding box is located relative to the page's printspace. The ordering is: 
            top margin, printspace, bottom margin, left margin, right margin. 
            See `htrflow.utils.layout.RegionLocation` for more details.
        3. The y-coordinate of the bounding box's top-left corner.

    Parameters
    ----------
    image : PILImage
        Input PILImage
    bboxes : Sequence[Bbox]
        Bounding boxes to be ordered.

    Returns
    -------
    list[int]
        A list of integers `index` where `index[i]` is the suggested reading order of the i:th bounding box.
    """
    # Estimate printspace of the image. Returns a bbox covering the main reading area
    printspace = estimate_printspace(np.array(image))

    # Check if the image is two-page by finding a middle line represented by very dark pixels
    is_twopage = check_twopage(np.array(image))

    def key(i: int):
        return (
            is_twopage and (bboxes[i].center.x > printspace.center.x),
            get_region_location(printspace, bboxes[i]).value,  
            bboxes[i].ymin,
        )

    return sorted(range(len(bboxes)), key=key)


# Code from ChatGPT
def sort_top_down_left_right(bboxes: Sequence[Bbox], split_x: float | None = None) -> list[int]:
    """Order bounding boxes using a simple heuristic.

    Automatically splits bounding boxes into 'left' and 'right' groups based
    on a guessed `split_x` (center x of all boxes if not provided).
    Within each group, boxes are ordered top-down (smallest y first).

    Drawback of this method is that lines on the margin may be merged into the adjacent lines

    Parameters
    ----------
    bboxes : Sequence[Bbox]
        Input bounding boxes
    split_x : float | None, optional
        split_x: Optional. If None, will guess by median center x of bboxes.

    Returns
    -------
    list[int]
        list of indices of the original bboxes in the new order
    """

    if len(bboxes) == 0:
        return []

    if split_x is None:
        centers_x = [(bbox.xmin + bbox.xmax) / 2 for bbox in bboxes]
        centers_x.sort()
        median_idx = len(centers_x) // 2
        split_x = centers_x[median_idx]

    left_indices = []
    right_indices = []

    for idx, bbox in enumerate(bboxes):
        center_x = (bbox.xmin + bbox.xmax) / 2
        if center_x < split_x:
            left_indices.append(idx)
        else:
            right_indices.append(idx)

    # Sort left side: top-down, then left-right
    left_sorted = sorted(left_indices, key=lambda i: (bboxes[i].ymin, bboxes[i].xmin))

    # Sort right side: top-down, then left-right
    right_sorted = sorted(right_indices, key=lambda i: (bboxes[i].ymin, bboxes[i].xmin))

    return left_sorted + right_sorted


def sort_page(page: Page, image: PILImage):
    region_bboxes = [region["bbox"] for region in page.regions]
    region_polygons = [region["polygon"] for region in page.regions]

    sorted_region_indices = sort_consider_margins(region_bboxes, image)

    page_region_bboxes    = [region_bboxes[i] for i in sorted_region_indices]
    page_region_polygons  = [region_polygons[i] for i in sorted_region_indices]
    page_region_lines     = []

    # Get region lines
    for region_idx in sorted_region_indices:
        region_line_objs = page.regions[region_idx]["lines"]

        region_line_bboxes = [line["bbox"] for line in region_line_objs]
        region_line_polygons = [line["polygon"] for line in region_line_objs]
        region_line_texts = [line["text"] for line in page.regions[region_idx]["lines"]]

        sorted_line_indices = sort_consider_margins(region_line_bboxes, image)

        region_line_bboxes      = [region_line_bboxes[i] for i in sorted_line_indices]
        region_line_polygons    = [region_line_polygons[i] for i in sorted_line_indices]
        region_line_texts       = [region_line_texts[i] for i in sorted_line_indices]

        region_lines = [Line(*tup) for tup in zip(region_line_bboxes, region_line_polygons, region_line_texts)]
        page_region_lines.append(region_lines)

    page_regions = [Region(*tup) for tup in zip(page_region_bboxes, page_region_polygons, page_region_lines)]

    page_lines = []
    for lines in page_region_lines:
        page_lines += lines

    return Page(regions=page_regions, lines=page_lines)


# Shift line bbox and polygon detected in cropped image to match larger image
def correct_line_bbox_coords(region_bbox: Bbox, line_bbox: Bbox):
    shift_x = region_bbox[0]
    shift_y = region_bbox[1]

    correct_bbox = Bbox(
        line_bbox[0] + shift_x,
        line_bbox[1] + shift_y,
        line_bbox[2] + shift_x,
        line_bbox[3] + shift_y
    )
    return correct_bbox


def correct_line_polygon_coords(region_bbox: Bbox, line_polygon: Polygon):
    shift_x = region_bbox[0]
    shift_y = region_bbox[1]

    correct_polygon = Polygon([(x + shift_x, y + shift_y) for (x, y) in line_polygon.boundary.coords])
    return correct_polygon


def merge_bboxes(bbox1: Bbox, bbox2: Bbox):
    """Merge two boxes into one."""
    x_min = min(bbox1[0], bbox2[0])
    y_min = min(bbox1[1], bbox2[1])
    x_max = max(bbox1[2], bbox2[2])
    y_max = max(bbox1[3], bbox2[3])
    return (x_min, y_min, x_max, y_max)


def coverage_ratio(box_small: Bbox, box_large: Bbox):
    """Compute how much of box_small is covered by box_large."""
    x1 = max(box_small[0], box_large[0])
    y1 = max(box_small[1], box_large[1])
    x2 = min(box_small[2], box_large[2])
    y2 = min(box_small[3], box_large[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area_small = (box_small[2] - box_small[0]) * (box_small[3] - box_small[1])

    if area_small == 0:
        return 0.0
    return inter_area / area_small


def merge_overlapping_bboxes(boxes: list[Bbox], iou_threshold=0.2, coverage_threshold=0.5) -> list[Bbox]:
    """Merge overlapping or covering boxes."""
    merged = []
    boxes = boxes.copy()

    while boxes:
        box = boxes.pop(0)
        has_merged = False

        for i, mbox in enumerate(merged):
            area_box = (box[2] - box[0]) * (box[3] - box[1])
            area_mbox = (mbox[2] - mbox[0]) * (mbox[3] - mbox[1])

            # Determine which box is smaller/larger
            if area_box < area_mbox:
                small, large = box, mbox
            else:
                small, large = mbox, box

            cov_ratio = coverage_ratio(small, large)

            if (compute_bbox_iou(box, mbox) > iou_threshold or cov_ratio >= coverage_threshold):
                merged[i] = merge_bboxes(box, mbox)
                has_merged = True
                break

        if not has_merged:
            merged.append(box)

    # Second pass to clean up new overlaps
    changed = True
    while changed:
        changed = False
        new_merged = []
        while merged:
            box = merged.pop(0)
            has_merged = False
            for i, mbox in enumerate(new_merged):
                area_box = (box[2] - box[0]) * (box[3] - box[1])
                area_mbox = (mbox[2] - mbox[0]) * (mbox[3] - mbox[1])

                if area_box < area_mbox:
                    small, large = box, mbox
                else:
                    small, large = mbox, box

                cov_ratio = coverage_ratio(small, large)

                if (compute_bbox_iou(box, mbox) > iou_threshold or cov_ratio >= coverage_threshold):
                    new_merged[i] = merge_bboxes(box, mbox)
                    has_merged = True
                    changed = True
                    break
            if not has_merged:
                new_merged.append(box)
        merged = new_merged

    merged_bboxes = [Bbox(*bbox) for bbox in merged]

    return merged_bboxes