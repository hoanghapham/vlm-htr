import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL.Image import Image
from src.data_processing.visual_tasks import bbox_xyxy_to_xywh
import numpy as np


def random_color():
    return (random.random(), random.random(), random.random())  # RGBA with transparency


def draw_bboxes_xyxy(
    image: Image, 
    bboxes: list[list|tuple], 
    fig_size: int = 15, 
    bbox_color: str = "red", 
    label_facecolor: str = "lightcoral",
    label_color: str = "black",
):
    image = image.convert("RGB")

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    ax.imshow(image)

    # Draw pred bboxes
    for idx, bbox in enumerate(bboxes):
        x, y, width, height = bbox_xyxy_to_xywh(bbox)
        rect = patches.Rectangle(
            (x, y), width, height,
            linewidth=2, edgecolor=bbox_color, facecolor='none', label=idx
        )
        ax.add_patch(rect)
        plt.text(
            x, y, 
            idx, 
            color = label_color, 
            fontsize = 8, 
            bbox = dict(facecolor=label_facecolor, alpha=1)
        )
    
    ax.axis('off')
    plt.show()


def draw_page_line_segments(
        image: Image, 
        regions_with_lines: list[dict], 
        show_region_bbox=True, 
        show_line_mask=True, 
        show_line_bbox=False, 
        show_polygon_idx=False,
        fig_size=15,
        number_size=8
    ):
    
    image = image.convert("RGB")
    # Show image
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    ax.imshow(image)
    ax.set_axis_off()

    lines_data = []
    for region in regions_with_lines:
        lines_data += region["lines"]

    # Color for lines
    line_colors = [random_color() for _ in range(len(lines_data))]

    if show_region_bbox:
        # Draw bbox for each large region
        for idx, region in enumerate(regions_with_lines):
            x, y, width, height = bbox_xyxy_to_xywh(region["bbox"])

            # bbox is not guaranteed to be a rectangle, and can actually have weird shapes
            rect = patches.Rectangle(
                (x, y), width, height,
                linewidth=2, edgecolor='r', facecolor='none', label="Bounding Box"
            )
            ax.add_patch(rect)
            plt.text(
                x, y, 
                idx, 
                color = "black", 
                fontsize = number_size, 
                bbox = dict(facecolor="lightcoral", alpha=1)
            )

    # Color lines
    if show_line_mask:
        for idx, line in enumerate(lines_data):
            seg_x = [x for (x, y) in line["polygon"]]
            seg_y = [y for (x, y) in line["polygon"]]
            ax.fill(
                seg_x, seg_y, 
                facecolor=line_colors[idx], 
                alpha=0.5, 
                edgecolor=line_colors[idx], 
                linewidth=2, label="Segmentation"
            )
            if show_polygon_idx:
                x = min(seg_x)
                y = min(seg_y)
                plt.text(
                    x, y, 
                    idx, 
                    color = "black", 
                    fontsize = number_size, 
                    bbox = dict(facecolor="lightcoral", alpha=1)
                )

    if show_line_bbox:
        # Construct bbox for each line
            
        for idx, line in enumerate(lines_data):
            x, y, width, height = bbox_xyxy_to_xywh(line["bbox"])

            rect = patches.Rectangle(
                (x, y), width, height,
                linewidth=2, edgecolor='r', facecolor='none', label="Bounding Box"
            )
            ax.add_patch(rect)


def draw_segment_masks(image: Image, masks: list[list[tuple | list]], fig_size=15):
    """Draw segmentation masks from provided polygons

    Parameters
    ----------
    image : Image
    masks : list[list[tuple  |  list]]
        List of masks, each mask is a list of (x, y) coordinates
    fig_size : int, optional
    """
    image = image.convert("RGB")
    # Show image
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    ax.imshow(image)
    ax.set_axis_off()

    # Color for lines
    line_colors = [random_color() for _ in range(len(masks))]

    for idx, poly in enumerate(masks):
        seg_x = [x for (x, y) in poly]
        seg_y = [y for (x, y) in poly]
        ax.fill(
            seg_x, seg_y, 
            facecolor=line_colors[idx], 
            alpha=0.5, 
            edgecolor=line_colors[idx], 
            linewidth=2, label="Segmentation"
        )


def draw_object_bbox_segment(image: Image, bbox: tuple, mask: list[tuple], fig_size=15):
    """Draw bbox and segmentation mask of one object

    Parameters
    ----------
    image : Image
    bbox : tuple
        bbox in xyxy format: (x1, y1, x2, y2)
    mask : list[tuple]
        list of (x, y) tuples representing the polygon coords
    figsize : int, optional
        Size of the figure, by default 15
    """
    image = image.convert("RGB")
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    ax.imshow(image)

    # Draw bbox
    x, y, width, height = bbox_xyxy_to_xywh(bbox)
    rect = patches.Rectangle(
        (x, y), width, height,
        linewidth=2, edgecolor="red", facecolor='none'
    )
    ax.add_patch(rect)
    
    # Draw mask
    mask_color = random_color()
    seg_x = [x for (x, y) in mask]
    seg_y = [y for (x, y) in mask]
    ax.fill(
        seg_x, seg_y, 
        facecolor=mask_color, 
        alpha=0.5, 
        edgecolor=mask_color, 
        linewidth=2, label="Segmentation"
    )

    ax.axis('off')
    plt.show()
