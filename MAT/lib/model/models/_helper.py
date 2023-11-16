import torch
from torch import Tensor
from torchvision.ops import clip_boxes_to_image, box_iou


def compute_iou(pred_box: Tensor, target_box: Tensor):
    """

    Args:
        pred_box: Tensor (N, 4) normalized box [[x1, y1, x2, y2]]
        target_box: Tensor (N, 4) normalized box [[x1, y1, x2, y2]]

    Returns:
        iou: Tensor (N, 1)
    """
    pred_box = clip_boxes_to_image(pred_box, size=(1, 1))
    pred_iou = box_iou(pred_box, target_box)
    iou = torch.diag(pred_iou)

    return iou.reshape(-1, 1)
