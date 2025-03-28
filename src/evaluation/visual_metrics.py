from sklearn.metrics import precision_recall_fscore_support
from shapely import Polygon, union_all
from htrflow.evaluate import Ratio


def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area

    return inter_area * 1.0 / union_area


def precision_recall_fscore(detections: list[list], annotations: list[list], iou_threshold=0.5):
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
                iou = compute_iou(d, a)
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

    precision_avg, recall_avg, f1_score_avg, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

    return precision_avg, recall_avg, f1_score_avg


def region_coverage(pred_polygons: list[Polygon], truth_polygons: list[Polygon]):
        truth = union_all(truth_polygons)
        pred = union_all(pred_polygons)
        return Ratio(truth.intersection(pred).area, truth.area)
