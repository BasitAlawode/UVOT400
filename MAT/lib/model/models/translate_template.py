import importlib
from typing import List, Union, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision.ops import box_convert


class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()

        self.pretrained_param = None

        self.register_buffer("pytorch_mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1))  # RGB
        self.register_buffer("pytorch_std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1))  # RGB

        self.cfg = args

        # build backbone
        backbone_module = importlib.import_module('lib.model.backbones')
        self.backbone = getattr(backbone_module, self.cfg.backbone.type)(self.cfg.backbone)

        # # build multi-head
        # self.cfg.head.in_channels = self.backbone.out_channels_list[-1]
        # head_module = importlib.import_module('lib.model.heads')
        # self.head = getattr(head_module, self.cfg.head.type)(self.cfg.head, self.cfg.head.output_size)
        #
        # # build criterion
        # criteria_module = importlib.import_module('lib.criteria')
        # self.criteria = [getattr(criteria_module, tp)(self.cfg.criterion) for tp in self.cfg.criterion.type]

    def forward(self, input_dict: Dict[str, Union[Tensor, Any]]):
        device = next(self.parameters()).device

        template_in: Tensor = self._pytorch_norm(input_dict['template_in'].to(device))
        search: Tensor = self._pytorch_norm(input_dict['search'].to(device))
        template_out: Tensor = self._pytorch_norm(input_dict['template_out'].to(device))
        # target_boxes: Tensor = input_dict['target'].to(device)

        bs = search.shape[0]

        loss, pred_out, mask, loss_x, pred_x = self.backbone(search, template_in, template_out)

        # template_in = torch.einsum('nchw->nhwc', template_in)
        # pred_out = torch.einsum('nchw->nhwc', pred_out)

        template_in = self._de_norm(template_in)
        template_out = self._de_norm(template_out)
        pred_out = self._de_norm(pred_out)
        search = self._de_norm(search)
        pred_x = self._de_norm(pred_x)

        # # visualize the mask
        # mask = mask.detach()
        # mask = mask.unsqueeze(-1).repeat(1, 1, self.backbone.patch_embed.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3)
        # mask = self.backbone.unpatchify(mask)  # 1 is removing, 0 is keeping
        # mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
        #
        # # masked image
        # t_masked = template_in * (1 - mask)
        # # MAE reconstruction pasted with visible patches
        # t_paste = z * (1 - mask) + trans_t * mask

        loss_dict = dict()
        metric_dict = dict()
        vis_dict = dict()
        total_loss = loss + loss_x
        loss_dict.update({'t': loss, 's': loss_x})

        vis_dict.update({'t_in': template_in, 't_out': template_out, 't_pred': pred_out})
        vis_dict.update({'s_in': search, 's_pred': pred_x})

        return total_loss, [loss_dict, metric_dict, vis_dict]

    def init(self, img, box):
        t = self.backbone(img, box=box)
        return t

    def track(self,
              img: Tensor
              ):
        ns, _, hs, ws = img.shape

        features: List[Tensor] = self.backbone(self._pytorch_norm(img))  # (N, C, H, W)
        feature = features[-1]  # (N, 768, 8, 12)

        f = feature[..., :8]  # (N, 768, 8, 8)
        f = F.max_pool2d(f, 8, 8).flatten(1)
        pred_box = self.head(f)

        outputs_coord = box_convert(pred_box, in_fmt='xyxy', out_fmt='cxcywh')

        pred_dict = dict()
        pred_dict['box'] = outputs_coord.squeeze().detach().cpu().numpy()
        pred_dict['score'] = 1
        pred_dict['visualize'] = [None, None]

        return pred_dict

    def _pytorch_norm(self, img):
        img = img.div(255.0)
        img = img.sub(self.pytorch_mean).div(self.pytorch_std)
        return img

    def _de_norm(self, x):
        x = torch.clip((x * self.pytorch_std + self.pytorch_mean), 0, 1)
        return x


def build_translate_template(args):
    model = Model(args)
    return model


if __name__ == '__main__':
    from config.cfg_translation import cfg

    net = build_translate_template(cfg.model)

    gt = torch.rand(2, 4)
    gt[:, :2] = gt[:, :2] - gt[:, 2:] * 0.5
    gt[:, 2:] = gt[:, :2] + gt[:, 2:]
    gt = torch.clip(gt, 0, 1)
    x = torch.rand(2, 3, 224, 224)
    z = torch.rand(2, 3, 112, 112)

    in_dict = {
        'search': x,
        'template_in': z,
        'template_out': z,
        'target': gt,
        'training': True,
    }

    out = net(in_dict)
    print(out)

    from ptflops import get_model_complexity_info


    def prepare_input(resolution):
        input_dict = {
            'search': x,
            'template_in': z,
            'template_out': z,
            'target': gt,
            'training': False,
        }

        return dict(input_dict=input_dict)


    flops, params = get_model_complexity_info(net,
                                              input_res=(x.shape, gt.shape),
                                              input_constructor=prepare_input,
                                              as_strings=True, print_per_layer_stat=False)
    print('      - Macs:  ' + flops)
    print('      - Params: ' + params)
