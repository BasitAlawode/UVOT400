import torch.nn as nn
from easydict import EasyDict as Edict


class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()

        self.cfg = Edict
        self.in_channels: int
        self.inter_channels: int
        self.feat_h: int
        self.feat_w: int

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def predict_box(self, *args, **kwargs):
        raise NotImplementedError

    def predict_score(self, *args, **kwargs):
        raise NotImplementedError
