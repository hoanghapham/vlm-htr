from typing import Sequence
from htrflow.utils.geometry import Bbox


# Code from https://github.com/AI-Riksarkivet/htrflow/blob/main/src/htrflow/postprocess/reading_order.py, with slight modifications

def order_regions(regions: Sequence, printspace: Bbox, is_twopage: bool = False):
    """Order regions according to their reading order

    This function estimates the reading order based on the following:
        1. Which page of the spread the region belongs to (if
            `is_twopage` is True)
        2. Where the region is located relative to the page's
            printspace. The ordering is: top margin, printspace,
            bottom margin, left margin, right margin. See
            `layout.RegionLocation` for more details.
        3. The y-coordinate of the region's top-left corner.

    This function can be used to order the top-level regions of a
    page, but is also suitable for ordering the lines within each
    region.

    Arguments:
        regions: Regions to be ordered.
        printspace: A bounding box around the page's printspace.
        is_twopage: Whether the page is a two-page spread.

    Returns:
        The input regions in reading order.
    """
    index = order_bboxes([region.bbox for region in regions], printspace, is_twopage)
    return [regions[i] for i in index]


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