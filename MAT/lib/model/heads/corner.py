import torch.nn as nn
import torch
import torch.nn.functional as F

from lib.model.heads._head import Head as BaseHead


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, freeze_bn=False):
    if freeze_bn:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            FrozenBatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()  # rsqrt(x): 1/sqrt(x), r: reciprocal
        bias = b - rm * scale
        return x * scale + bias


class Corner(BaseHead):
    """ Corner Predictor module"""

    def __init__(self, args, freeze_bn=False):
        super(Corner, self).__init__()

        self.cfg = args
        self.in_channels = self.cfg.in_channels
        self.inter_channels = self.cfg.inter_channels
        self.feat_h, self.feat_w = self.cfg.search_size

        self.stride = self.cfg.stride
        assert self.feat_h == self.feat_w, "not support non-square feature map"
        self.feat_sz = self.feat_h
        self.img_sz = self.feat_sz * self.stride

        '''top-left corner'''
        self.conv1_tl = conv(self.in_channels, self.inter_channels, freeze_bn=freeze_bn)
        self.conv2_tl = conv(self.inter_channels, self.inter_channels // 2, freeze_bn=freeze_bn)
        self.conv3_tl = conv(self.inter_channels // 2, self.inter_channels // 4, freeze_bn=freeze_bn)
        self.conv4_tl = conv(self.inter_channels // 4, self.inter_channels // 8, freeze_bn=freeze_bn)
        self.conv5_tl = nn.Conv2d(self.inter_channels // 8, 1, kernel_size=1)

        '''bottom-right corner'''
        self.conv1_br = conv(self.in_channels, self.inter_channels, freeze_bn=freeze_bn)
        self.conv2_br = conv(self.inter_channels, self.inter_channels // 2, freeze_bn=freeze_bn)
        self.conv3_br = conv(self.inter_channels // 2, self.inter_channels // 4, freeze_bn=freeze_bn)
        self.conv4_br = conv(self.inter_channels // 4, self.inter_channels // 8, freeze_bn=freeze_bn)
        self.conv5_br = nn.Conv2d(self.inter_channels // 8, 1, kernel_size=1)

        '''about coordinates and indexs'''
        # with torch.no_grad():
        indice = torch.arange(0, self.feat_sz).view(-1, 1) * self.stride
        # generate mesh-grid
        coord_x = indice.repeat((self.feat_sz, 1)).view((self.feat_sz * self.feat_sz,)).float()
        coord_y = indice.repeat((1, self.feat_sz)).view((self.feat_sz * self.feat_sz,)).float()

        self.register_buffer("coord_x", coord_x)
        self.register_buffer("coord_y", coord_y)  # RGB

    def forward(self, x, return_dist=True, softmax=True):
        """ Forward pass with input x. """
        score_map_tl, score_map_br = self.get_score_map(x)
        if return_dist:
            coorx_tl, coory_tl, prob_vec_tl = self.soft_argmax(score_map_tl, return_dist=True, softmax=softmax)
            coorx_br, coory_br, prob_vec_br = self.soft_argmax(score_map_br, return_dist=True, softmax=softmax)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz, prob_vec_tl, prob_vec_br
        else:
            coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
            coorx_br, coory_br = self.soft_argmax(score_map_br)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz

    def get_score_map(self, x):
        # top-left branch
        x_tl1 = self.conv1_tl(x)
        x_tl2 = self.conv2_tl(x_tl1)
        x_tl3 = self.conv3_tl(x_tl2)
        x_tl4 = self.conv4_tl(x_tl3)
        score_map_tl = self.conv5_tl(x_tl4)

        # bottom-right branch
        x_br1 = self.conv1_br(x)
        x_br2 = self.conv2_br(x_br1)
        x_br3 = self.conv3_br(x_br2)
        x_br4 = self.conv4_br(x_br3)
        score_map_br = self.conv5_br(x_br4)
        return score_map_tl, score_map_br

    def soft_argmax(self, score_map, return_dist=False, softmax=True):
        """ get soft-argmax coordinate for a given heatmap """
        score_vec = score_map.view((-1, self.feat_sz * self.feat_sz))  # (batch, feat_sz * feat_sz)
        prob_vec = nn.functional.softmax(score_vec, dim=1)
        exp_x = torch.sum((self.coord_x * prob_vec), dim=1)
        exp_y = torch.sum((self.coord_y * prob_vec), dim=1)
        if return_dist:
            if softmax:
                return exp_x, exp_y, prob_vec
            else:
                return exp_x, exp_y, score_vec
        else:
            return exp_x, exp_y
