import numpy as np
from typing import Tuple


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Return intersection-over-union (Jaccard index) between two sets of boxes.

    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        boxes1 (np.ndarray[N, 4]): first set of boxes
        boxes2 (np.ndarray[M, 4]): second set of boxes

    Returns:
        np.ndarray[N, M]: the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    boxes1 = np.clip(boxes1, 0, 9999)
    boxes2 = np.clip(boxes2, 0, 9999)

    inter, union = _box_inter_union(boxes1, boxes2)
    iou = inter / union
    return iou


def box_area(boxes: np.ndarray) -> np.ndarray:
    """
    Computes the area of a set of bounding boxes, which are specified by their
    (x1, y1, x2, y2) coordinates.

    Args:
        boxes (np.ndarray[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Returns:
        np.ndarray[N]: the area for each box
    """

    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def _box_inter_union(boxes1: np.ndarray, boxes2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = np.clip(rb - lt, 0, 9999)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    return inter, union
