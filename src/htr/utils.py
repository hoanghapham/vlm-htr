import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

import numpy as np
from typing import Sequence
from PIL.Image import Image as PILImage
from htrflow.utils.geometry import Bbox
from htrflow.utils.layout import estimate_printspace, is_twopage as check_twopage, get_region_location


# Code from https://github.com/AI-Riksarkivet/htrflow/blob/main/src/htrflow/postprocess/reading_order.py, with modifications

def sort_consider_margin(bboxes: Sequence[Bbox], image: PILImage) -> list[int]:
    """Order bounding boxes with respect to printspace, and consider margin.

    This function estimates the reading order based on the following:
        1. Which page of the spread the bounding box belongs to (if `is_twopage` is True)
        2. Where the bounding box is located relative to the page's printspace. The ordering is: 
            top margin, printspace, bottom margin, left margin, right margin. 
            See `htrflow.utils.layout.RegionLocation` for more details.
        3. The y-coordinate of the bounding box's top-left corner.

    TODO: This causes poor performance in traditional region_od__line_seg__ocr pipeline. 
    Investigate when have more time.

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
