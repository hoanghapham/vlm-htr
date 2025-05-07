import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

import numpy as np
from typing import Sequence
from PIL.Image import Image as PILImage
from htrflow.utils.geometry import Bbox
from htrflow.evaluate import Ratio
from htrflow.utils.layout import estimate_printspace, is_twopage as check_twopage, get_region_location

from src.file_tools import read_json_file


def read_img_metrics(
    img_metric_path: str | Path, 
    cer_list: list, 
    wer_list: list, 
    bow_hits_list: list, 
    bow_extras_list: list
) -> tuple[list, list, list, list]:
    img_metric = read_json_file(img_metric_path)
    cer_list.append(Ratio(*img_metric["cer"]["str"].split("/")))
    wer_list.append(Ratio(*img_metric["wer"]["str"].split("/")))
    bow_hits_list.append(Ratio(*img_metric["bow_hits"]["str"].split("/")))
    bow_extras_list.append(Ratio(*img_metric["bow_extras"]["str"].split("/")))

    return cer_list, wer_list, bow_hits_list, bow_extras_list


# Code from https://github.com/AI-Riksarkivet/htrflow/blob/main/src/htrflow/postprocess/reading_order.py, with slight modifications


def sort_bboxes(image: PILImage, bboxes: Sequence[Bbox]) -> list[int]:
    """Order bounding boxes with respect to printspace

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

    printspace = estimate_printspace(np.array(image))
    is_twopage = check_twopage(image)

    def key(i: int):
        return (
            is_twopage and (bboxes[i].center.x > printspace.center.x),
            # This causes a weird issue: the first two lines of a region are ordered last
            get_region_location(printspace, bboxes[i]).value,  
            bboxes[i].ymin,
        )

    return sorted(range(len(bboxes)), key=key)


# Code from ChatGPT
def sort_top_down_left_right(bboxes: Sequence[Bbox], split_x: float | None = None):
    """Order bounding boxes topdown-left right.

    Automatically splits bounding boxes into 'left' and 'right' groups based
    on a guessed `split_x` (center x of all boxes if not provided).
    Within each group, boxes are ordered top-down (smallest y first), then left-right (smallest x).

    Parameters
    ----------
    bboxes : Sequence[Bbox]
        Input bounding boxes
    split_x : float | None, optional
        split_x: Optional. If None, will guess by median center x of bboxes.

    Returns
    -------
    _type_
        _description_
    """

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
