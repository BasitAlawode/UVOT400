import torch
from easydict import EasyDict as Edict
from torch import nn


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.register_buffer("pytorch_mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1))  # RGB
        self.register_buffer("pytorch_std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1))  # RGB

        self.pretrained_param = None

        self.cfg: Edict
        self.backbone = nn.Module
        self.neck = nn.Module
        self.head = nn.Module
        self.criteria = nn.Module

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def init(self, *args, **kwargs):
        raise NotImplementedError

    def track(self, *args, **kwargs):
        raise NotImplementedError

    def get_backbone_feature(self, *args, **kwargs):
        raise NotImplementedError

    def _imagenet_norm(self, img):
        img = img.div(255.0)
        img = img.sub(self.pytorch_mean).div(self.pytorch_std)
        return img
