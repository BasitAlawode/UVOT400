import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_convert, box_iou, generalized_box_iou
import math
from typing import List, Optional
from torchvision.ops import roi_align


class SegBCE(object):
    def __init__(self, cfg):
        self.bce = nn.BCEWithLogitsLoss()

    def __call__(self, pred_mask: torch.Tensor, target_mask: torch.Tensor, target_box: Optional[torch.Tensor] = None):
        """

        Args:
            pred_mask:
            target_mask:
            target_box: norm[x, y, x, y]

        Returns:

        """

        assert pred_mask.shape == target_mask.shape, "pred_mask and target_mask must have same shape"

        # ignore the sample who has -1 target
        select = target_mask[:, 0, 0, 0] >= 0

        if select.sum().item() > 0:
            # ns, cs, hs, ws = pred_mask.shape
            # assert hs == ws, "not support rectangle, only support square"
            # coord = target_box.detach().clone() * hs
            #
            # rois = torch.cat((torch.arange(ns, dtype=torch.int32, device=target_box.device).float().unsqueeze(1),
            #                   coord), dim=1)
            #
            # pred_mask = roi_align(pred_mask, rois, output_size=128, sampling_ratio=0)
            # target_mask = roi_align(target_mask, rois, output_size=128, sampling_ratio=0)

            loss = self.bce(pred_mask[select], target_mask[select])

            tmp = (pred_mask.detach() > 0.0).float() + target_mask.detach()
            miou = (tmp == 2).sum() / (tmp > 0).sum()

            return [loss, {'bce': loss.item()}], [{'miou': miou.mean().item()}]

        else:
            return [None], [None]

