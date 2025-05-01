from typing import Sequence
from htrflow.utils.geometry import Bbox


# Code from https://github.com/AI-Riksarkivet/htrflow/blob/main/src/htrflow/postprocess/reading_order.py, with slight modifications

def order_bboxes(bboxes: Sequence[Bbox], printspace: Bbox, is_twopage: bool):
    """Order bounding boxes with respect to printspace

    This function estimates the reading order based on the following:
        1. Which page of the spread the bounding box belongs to (if
            `is_twopage` is True)
        2. Where the bounding box is located relative to the page's
            printspace. The ordering is: top margin, printspace,
            bottom margin, left margin, right margin. See
            `layout.RegionLocation` for more details.
        3. The y-coordinate of the bounding box's top-left corner.

    Arguments:
        bboxes: Bounding boxes to be ordered.
        printspace: A bounding box around the page's printspace.
        is_twopage: Whether the page is a two-page spread.

    Returns:
        A list of integers `index` where `index[i]` is the suggested
        reading order of the i:th bounding box.
    """

    def key(i: int):
        return (
            is_twopage and (bboxes[i].center.x > printspace.center.x),
            # get_region_location(printspace, bboxes[i]).value,  # This causes a weird issue: the first two lines of a region are ordered last
            bboxes[i].ymin,
        )

    return sorted(range(len(bboxes)), key=key)


# Code from ChatGPT
def topdown_left_right(bboxes: Sequence[Bbox], split_x: float | None = None):
    """Order bounding boxes topdown-left right.

    Automatically splits bounding boxes into 'left' and 'right' groups based
    on a guessed `split_x` (center x of all boxes if not provided).
    Within each group, boxes are ordered top-down (smallest y first), then left-right (smallest x).

    Arguments:
        bboxes: Input bounding boxes
        split_x: Optional. If None, will guess by median center x of bboxes.

    Returns:
        A sorted index.
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
