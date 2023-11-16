from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.model.necks._neck import Neck as BaseNeck


class DWCorr(BaseNeck):
    def __init__(self, args):
        super(DWCorr, self).__init__()

        self.cfg = args
        self.in_channels_list = self.cfg.in_channels_list
        self.inter_channels = self.cfg.inter_channels
        self.feat_h, self.feat_w = self.cfg.search_size
        self.feat_t_h, self.feat_t_w = self.cfg.template_size

        self.search_token_length = np.prod(self.cfg.search_size).astype(np.int32)

        self.projector = nn.Linear(self.in_channels_list[-1], self.inter_channels)

    def forward(self,
                search_features: torch.Tensor,
                template_features: torch.Tensor,
                *args, **kwargs):
        """

        Args:
            template_features: Tensor (N, HW, C)
            search_features: Tensor (N, HW, C)

        Returns:
            real_feat: Tensor (N, C, H, W)

        """
        s_token = self.parse_backbone_feature(search_features)  # (N, C, HW)
        t_token = self.parse_backbone_feature(template_features)  # (N, C, HW)

        ns, cs, _ = s_token.shape

        s_feat = s_token.reshape(ns, -1, self.feat_h, self.feat_w)  # (N, C, H, W)
        t_feat = t_token.reshape(ns, -1, self.feat_t_h, self.feat_t_w)  # (N, C, H, W)

        # ----------- use real template -------------------------
        kernel_sz = 7
        t_feat = F.interpolate(t_feat, (kernel_sz, kernel_sz), mode='bilinear', align_corners=True)
        s_feat = s_feat.reshape(1, -1, self.feat_h, self.feat_w)
        t_feat = t_feat.reshape(-1, 1, kernel_sz, kernel_sz)
        out_feat = F.conv2d(s_feat, t_feat, groups=s_feat.shape[1], padding=kernel_sz//2)

        real_feat = out_feat.reshape(ns, -1, self.feat_h, self.feat_w)

        return real_feat

    def parse_backbone_feature(self, backbone_feature: torch.Tensor):

        token = backbone_feature
        token = self.projector(token).permute(0, 2, 1)  # (N, HW, C) -> (N, C, HW)

        return token


if __name__ == '__main__':
    from config.cfg_translation_track import cfg

    cfg = cfg.model.neck

    cfg.in_channels_list = [1024]
    cfg.inter_channels = 256
    cfg.search_size = [16, 16]
    cfg.template_size = [8, 8]

    net = DWCorr(cfg)

    x = torch.ones(1, cfg.search_size[0]*cfg.search_size[1], cfg.in_channels_list[0])
    z = torch.ones(1, cfg.template_size[0]*cfg.template_size[1], cfg.in_channels_list[0])

    out = net(x, z)
    [print(tt[0].shape) for tt in out if tt[0] is not None]

    from ptflops import get_model_complexity_info


    def prepare_input(resolution):
        input_tuple = {
            'template_features': z,
            'search_features': x,
        }

        return input_tuple


    flops, params = get_model_complexity_info(net,
                                              input_res=(None,),
                                              input_constructor=prepare_input,
                                              as_strings=True, print_per_layer_stat=False)
    print('      - Macs:  ' + flops)
    print('      - Params: ' + params)
