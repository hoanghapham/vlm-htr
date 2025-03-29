import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from src.data_processing.visual_tasks import bbox_xyxy_to_xywh


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


def show_poliskammare_img(image_path: str, xml_path: str, show_large_bbox=False, show_line_bbox=False, fig_size=15):

    # Get data
    content = parse_pagexml_file(xml_path)
    region_data = content.get_regions()
    line_data = content.get_lines()


    # Show image
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = Image.open(image_path)
    ax.imshow(image)


    # Color for lines
    line_colors = [random_color() for _ in range(len(line_data))]

    if show_large_bbox:
        # Draw bbox for each large region
        for idx, region in enumerate(region_data):
            bbox = region.coords.points

            # bbox is not guaranteed to be a rectangle, and can actually be of weird shapes
            rect = patches.Polygon(
                bbox,
                linewidth=2, edgecolor='r', facecolor='none', label="Bounding Box"
            )
            ax.add_patch(rect)


    # Color lines
    for idx, poly in enumerate(line_data):
        seg_x = [x for (x, y) in poly.coords.points]
        seg_y = [y for (x, y) in poly.coords.points]
        ax.fill(
            seg_x, seg_y, 
            facecolor=line_colors[idx], 
            alpha=0.5, 
            edgecolor=line_colors[idx], 
            linewidth=2, label="Segmentation"
        )


    if show_line_bbox:
        # Construct bbox for each line
        for idx, region in enumerate(region_data):
            region_lines = region.get_lines()
            
            for idx, poly in enumerate(region_lines):
                anchor_x, anchor_y, width, height = construct_line_bbox(poly)

                rect = patches.Rectangle(
                    (anchor_x, anchor_y), width, height,
                    linewidth=2, edgecolor='r', facecolor='none', label="Bounding Box"
                )
                ax.add_patch(rect)
