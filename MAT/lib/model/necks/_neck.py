from typing import Optional, List

import torch.nn as nn
from easydict import EasyDict as Edict


class Neck(nn.Module):
    def __init__(self):
        super(Neck, self).__init__()

        self.cfg = Edict
        self.in_channels_list: List[int]
        self.inter_channels: int
        self.feat_h: int
        self.feat_w: int

        self.projector: Optional[nn.Module] = None

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def track(self, *args, **kwargs):
        raise NotImplementedError

    def parse_backbone_feature(self, *args, **kwargs):
        raise NotImplementedError
