import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss


def gen_gt_conf_map(bboxes, patch_size=320, stride=16):
    heatmap_size = patch_size // stride

    bs = bboxes.shape[0]
    conf_score_map = torch.zeros(bs, heatmap_size, heatmap_size)

    bbox = bboxes * heatmap_size
    w, h = bbox[:, 2], bboxes[:, 3]

    classes = torch.arange(bs).reshape((bs, 1)).long()
    centers_x = (bbox[:, 0:1] + w / 2).round().long()
    centers_y = (bbox[:, 1:2] + h / 2).round().long()

    conf_score_map[classes, centers_x, centers_y] = 1
    return conf_score_map

# if __name__ == '__main__':
#     a = torch.sigmoid(torch.randn((16, 4)))
#     b = torch.sigmoid(torch.randn((16, 20, 20)))
#     print('input', a.shape)
#     print('pred', b.shape)
#     result = gen_gt_conf_map(a)
#     print('output', result.flatten(1))
#     target = result.flatten(1)
#     print('pred2', b.flatten(1))
#     pre_tensor = b.flatten(1)
#
#     loss_func = BCEWithLogitsLoss()
#     loss = loss_func(pre_tensor, target)
#     print(loss)
