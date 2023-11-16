import torch
import torch.nn.functional as F
from torchvision.ops import box_iou, generalized_box_iou
from typing import List
from lib.model.models._helper import compute_iou


class DETR(object):
    def __init__(self, cfg):
        self.alpha_giou = cfg.alpha_giou
        self.alpha_l1 = cfg.alpha_l1
        self.alpha_conf = cfg.alpha_conf

    @staticmethod
    def generate_target(predictions, target_boxes):
        """

        Args:
            predictions: pred_box: Tensor (N, 4) [x y x y]
            target_boxes: Tensor (N, 4) [x y x y]

        Returns:

        """
        targets = [target_boxes, compute_iou(predictions.detach(), target_boxes.detach())]
        return targets

    def __call__(self, predictions: List[torch.Tensor], targets: List[torch.Tensor], id=None):
        assert len(predictions) == 2, "predictions: must be normalized [pred_box, pred_conf]"
        assert len(targets) == 2, "targets: must be normalized [target_box, target_iou, loss_mask]"

        pred_box, pred_conf = predictions
        target_box, target_iou = targets

        loss_giou, loss_l1, iou = loss_box(pred_box, target_box.detach(), reduction='mean')
        if pred_conf is None:
            loss_conf = torch.tensor(0.0, device=target_box.device)
        else:
            loss_conf = F.mse_loss(pred_conf.reshape(-1), target_iou.reshape(-1).detach(), reduction='mean')

        loss = (self.alpha_giou * loss_giou
                + self.alpha_l1 * loss_l1
                + self.alpha_conf * loss_conf)

        if id is not None:
            return [loss,
                    {f'giou_{id}': loss_giou.item()},
                    {f'l1_{id}': loss_l1.item()},
                    {f'conf_{id}': loss_conf.item()}
                    ], \
                   [{f'miou_{id}': iou.mean().item()}]
        else:
            return [loss,
                    {'giou': loss_giou.item()},
                    {'l1': loss_l1.item()},
                    {'conf': loss_conf.item()}
                    ], \
                   [{'miou': iou.mean().item()}]


class RPN(object):
    def __init__(self, cfg):
        self.alpha_coord = 1#cfg.alpha_coord
        self.alpha_conf = 1#cfg.alpha_conf

    @staticmethod
    def generate_target(predictions, target_boxes):
        """

        Args:
            predictions: pred_box: Tensor (N, H, W, 4)
            target_boxes: Tensor (N, 4) [x y x y]

        Returns:
            targets:

        """
        ns, feat_h, feat_w, _ = predictions.shape
        tmp_target_boxes = target_boxes.unsqueeze(1).expand(-1, feat_h * feat_w, -1)  # (N, HW, 4)
        tmp_predictions = predictions.reshape(ns, -1, 4)  # (N, HW, 4)

        ious = compute_iou(tmp_predictions.detach().reshape(-1, 4),
                           tmp_target_boxes.detach().reshape(-1, 4))  # (NHW, 1)

        target = target_boxes.unsqueeze(1).unsqueeze(1).expand(-1, feat_h, feat_w, -1)  # (N, H, W, 4)
        ious = ious.reshape(ns, feat_h, feat_w, 1)  # (N, H, W, 1)

        loss_mask = torch.zeros((ns, feat_h, feat_w, 1), device=predictions.device)
        for i in range(ns):
            x1, y1, x2, y2 = target_boxes[i]
            loss_mask[i,
                      torch.round(y1 * feat_h).int():torch.round(y2 * feat_h).int(),
                      torch.round(x1 * feat_h).int():torch.round(x2 * feat_h).int()] = 1

        targets = [target, ious, loss_mask]
        return targets

    def __call__(self, predictions: List[torch.Tensor], targets: List[torch.Tensor]):
        """

        Args:
            predictions: pred_box: Tensor (N, H, W, 4), pred_conf: Tensor (N, H, W, 1)
            targets: target_box: Tensor (N, H, W, 4) [x y x y], target_iou: Tensor (N, H, W, 1)
                    loss_mask: Tensor (N, H, W, 1)
        Returns:

        """
        assert len(predictions) == 2, "predictions: must be normalized [pred_box, pred_conf]"
        assert len(targets) == 3, "targets: must be normalized [target_box, target_iou, loss_mask]"

        pred_box, pred_conf = predictions
        target_box, target_iou, loss_mask = targets

        n = loss_mask.sum()
        loss_coord = (F.mse_loss(pred_box, target_box, reduction='none') * loss_mask).sum() / n
        loss_conf = (F.mse_loss(pred_conf, target_iou, reduction='none') * loss_mask).sum() / n

        iou = (target_iou * loss_mask).sum() / n

        loss = (self.alpha_coord * loss_coord
                + self.alpha_conf * loss_conf)

        return [loss,
                {'box': loss_coord.item()},
                {'conf': loss_conf.item()}
                ], \
               [{'miou': iou.item()}]


def loss_box(pred, target, reduction='mean'):
    """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
       targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
       The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
    params:
        pred_box: [B 4]  [x y x y]
        target_box: [B 4]  [x y x y]
    return:
        loss_giou
        loss_bbox
    """

    if reduction == 'mean':
        loss_l1 = F.l1_loss(pred, target, reduction='mean')

        try:
            loss_iou = (1 - torch.diag(generalized_box_iou(pred, target))).mean()
            miou = torch.diag(box_iou(pred, target))
        except:
            loss_iou, miou = torch.tensor(0.0).to(pred.device), torch.zeros(pred.shape[0]).to(pred.device)

    else:
        loss_l1 = F.l1_loss(pred, target, reduction='none')

        try:
            loss_iou = (1 - torch.diag(generalized_box_iou(pred, target)))
            miou = torch.diag(box_iou(pred, target))
        except:
            loss_iou, miou = torch.zeros(pred.shape[0]).to(pred.device), torch.zeros(pred.shape[0]).to(pred.device)

    return loss_iou, loss_l1, miou
