import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from skimage.segmentation import find_boundaries
from scipy.optimize import linear_sum_assignment

from shapely import union_all
from shapely.geometry import Polygon
from htrflow.evaluate import Ratio
from htrflow.utils.geometry import Bbox 
from PIL import Image, ImageDraw

from src.evaluation.utils import Ratio


def compute_bbox_iou(bbox1: Bbox, bbox: Bbox):
    x1, y1, x2, y2 = bbox1
    x1g, y1g, x2g, y2g = bbox

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    bbox1_area = (x2 - x1) * (y2 - y1)
    bbox_area = (x2g - x1g) * (y2g - y1g)
    union_area = bbox1_area + bbox_area - inter_area

    return inter_area * 1.0 / union_area


def compute_bbox_precision_recall_fscore(detections: list[list], annotations: list[list], iou_threshold=0.5):
    assert len(detections) == len(annotations), "detections & annotations length mismatch"
    y_true = []
    y_pred = []

    for det, ann in zip(detections, annotations):
        true_pos = 0
        false_pos = 0
        # Initiate false_neg as total number of boxes in annotation
        # No matching bbox means that our pred_polygons all missed the annotated box
        false_neg = len(ann)

        # for each box in the annotation:
        for a in ann:
            matched = False

            # for each box in the detection
            # check if there is at least 1 box in the detection that matches the current annotated box
            # if yes, break
            for d in det:
                iou = compute_bbox_iou(d, a)
                if iou >= iou_threshold:
                    matched = True
                    break
            
            # If found a matched pred box for this ann box, increase true positive by 1
            # Decrease false negative by 1, because this current ann box was matched
            # max true_pos = Number of annotated boxes
            if matched:
                true_pos += 1
                false_neg -= 1
            
            # If found no match, increase false positive by 1
            # Max false positive = number of annotated box
            else:
                false_pos += 1

        # Length of y_true = (Number of annotated boxes) + false positive
        y_true.extend([1] * len(ann) + [0] * false_pos)

        # Length of y_pred = (number of annotated box that were matched) + 
        #   (Number of pred box that missed the mark) + (number of annotated boxes that were not matched)
        y_pred.extend([1] * true_pos + [0] * (false_pos + false_neg))

    precision_avg, recall_avg, f1_score_avg, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")

    return precision_avg, recall_avg, f1_score_avg


def compute_polygons_region_coverage(pred_polygons: list[Polygon], truth_polygons: list[Polygon]):
    truth = union_all(truth_polygons)
    pred = union_all(pred_polygons)
    return Ratio(truth.intersection(pred).area, truth.area)


def compute_masks_region_coverage(pred_mask, truth_mask):
    intersection = np.logical_and(pred_mask, truth_mask)
    intersection_area = np.sum(intersection)
    
    truth_area = np.sum(truth_mask)
    region_coverage = float(Ratio(intersection_area, truth_area))
    
    return region_coverage


# Segmentation metrics
# From ChatGPT
def polygon_to_binary_mask(polygon: list[tuple[int, int]], image_size):
    """
    Convert polygon to binary mask.
    
    Args:
        polygon (list of [x, y]): The polygon coordinates.
        image_size (tuple): (width, height) of the output mask.
    
    Returns:
        np.ndarray: Binary mask of shape (height, width)
    """
    mask_img = Image.new("L", image_size, 0)
    polygon_tuples = [tuple(p) for p in polygon]
    ImageDraw.Draw(mask_img).polygon(polygon_tuples, outline=1, fill=1)
    return np.array(mask_img, dtype=bool)


def compute_mask_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)
    return intersection.sum() / (union.sum() + 1e-10)


def compute_dice(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask)
    return 2 * intersection.sum() / (pred_mask.sum() + gt_mask.sum() + 1e-10)


def compute_pixel_accuracy(pred_mask, gt_mask):
    tp = np.logical_and(pred_mask, gt_mask).sum()
    tn = np.logical_and(~pred_mask, ~gt_mask).sum()
    total = pred_mask.size
    return (tp + tn) / total


def compute_mean_pixel_accuracy(pred_mask, gt_mask):
    tp = np.logical_and(pred_mask, gt_mask).sum()
    tn = np.logical_and(~pred_mask, ~gt_mask).sum()
    fg_acc = tp / (gt_mask.sum() + 1e-10)
    bg_acc = tn / ((~gt_mask).sum() + 1e-10)
    return (fg_acc + bg_acc) / 2


def compute_boundary_f1(pred_mask, gt_mask):
    pred_boundary = find_boundaries(pred_mask, mode="thick")
    gt_boundary = find_boundaries(gt_mask, mode="thick")
    intersection = np.logical_and(pred_boundary, gt_boundary).sum()
    precision = intersection / (pred_boundary.sum() + 1e-10)
    recall = intersection / (gt_boundary.sum() + 1e-10)
    return 2 * precision * recall / (precision + recall + 1e-10)


def compute_seg_metrics(pred_mask, gt_mask):
    return {
        "iou": compute_mask_iou(pred_mask, gt_mask),
        "dice": compute_dice(pred_mask, gt_mask),
        "pixel_accuracy": compute_pixel_accuracy(pred_mask, gt_mask),
        "mean_pixel_accuracy": compute_mean_pixel_accuracy(pred_mask, gt_mask),
        "boundary_f1": compute_boundary_f1(pred_mask, gt_mask),
        "region_coverage": compute_masks_region_coverage(pred_mask, gt_mask)
    }


def compute_iou_matrix(pred_polygons, gt_polygons, image_size):
    pred_masks = [polygon_to_binary_mask(p, image_size) for p in pred_polygons]
    gt_masks = [polygon_to_binary_mask(g, image_size) for g in gt_polygons]

    iou_matrix = np.zeros((len(gt_masks), len(pred_masks)))
    for i, g_mask in enumerate(gt_masks):
        for j, p_mask in enumerate(pred_masks):
            iou_matrix[i, j] = compute_mask_iou(p_mask, g_mask)

    return iou_matrix, pred_masks, gt_masks


# Match predicted masks with groundtruth masks, then calculate metrics
def match_and_evaluate(pred_polygons, gt_polygons, image_size, iou_threshold=0.5):
    iou_matrix, pred_masks, gt_masks = compute_iou_matrix(pred_polygons, gt_polygons, image_size)

    # Use Hungarian Algorithm (maximize IoU = minimize -IoU)
    gt_indices, pred_indices = linear_sum_assignment(-iou_matrix)

    results = []
    matched_preds = set()
    matched_gts = set()

    for gt_idx, pred_idx in zip(gt_indices, pred_indices):
        iou = iou_matrix[gt_idx, pred_idx]
        pred_mask = pred_masks[pred_idx]
        gt_mask = gt_masks[gt_idx]

        if iou >= iou_threshold:
            metrics = compute_seg_metrics(pred_mask, gt_mask)
            result = {
                "pair": (gt_idx, pred_idx),
                "metrics": metrics
            }
            results.append(result)
            matched_preds.add(pred_idx)
            matched_gts.add(gt_idx)

    # Unmatched predictions = false positives
    unmatched_preds = [idx for idx in range(len(pred_masks)) if idx not in matched_preds]
    unmatched_gts = [idx for idx in range(len(gt_masks)) if idx not in matched_gts]

    return {
        "matched": results,
        "unmatched_preds": unmatched_preds,
        "unmatched_gts": unmatched_gts,
        "num_preds": len(pred_masks),
        "num_gts": len(gt_masks)
    }
