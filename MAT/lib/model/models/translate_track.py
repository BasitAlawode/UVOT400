import importlib
from typing import Union, Dict, Any

import torch
from torch import Tensor
from torchvision.ops import box_convert

from MAT.lib.model.models._model import Model as BaseModel


class Model(BaseModel):

    def __init__(self, args):
        super(Model, self).__init__()

        self.pretrained_param = None

        self.cfg = args

        # build backbone
        backbone_module = importlib.import_module('MAT.lib.model.backbones')
        self.backbone = getattr(backbone_module, self.cfg.backbone.type)(self.cfg.backbone)
        self.cfg.backbone.out_stride = self.backbone.out_stride

        # build neck
        neck_module = importlib.import_module('MAT.lib.model.necks')
        self.cfg.neck.search_size = [sz // self.cfg.backbone.out_stride for sz in self.cfg.backbone.search_size]
        self.cfg.neck.template_size = [sz // self.cfg.backbone.out_stride for sz in self.cfg.backbone.template_size]
        self.cfg.neck.in_channels_list = [c for c in self.backbone.out_channels_list]
        self.neck = getattr(neck_module, self.cfg.neck.type)(self.cfg.neck)

        # build head
        head_module = importlib.import_module('MAT.lib.model.heads')
        self.cfg.head.search_size = self.cfg.neck.search_size
        self.cfg.head.stride = self.cfg.backbone.out_stride
        self.head = getattr(head_module, self.cfg.head.type)(self.cfg.head)

        # build criterion
        criteria_module = importlib.import_module('MAT.lib.criteria')
        self.criteria = getattr(criteria_module, self.cfg.criterion.type)(self.cfg.criterion)

    def forward(self, input_dict: Dict[str, Union[Tensor, Any]]):

        device = next(self.parameters()).device

        images: Tensor = input_dict['search'].to(device)
        templates: Tensor = input_dict['template'].to(device)
        target_boxes: Tensor = input_dict['target'].to(device)

        # ----------- backbone feature -------------------------
        s_feat, t_feat = self.get_backbone_feature(images, templates)

        # ----------- fusion -------------------------
        feat = self.neck(s_feat, t_feat)

        # ----------- predict -------------------------
        pred_boxes, score_lt, score_br = self.head(feat)

        # ----------- compute loss -------------------------
        loss_dict = dict()
        metric_dict = dict()

        losses, metrics = self.criteria([pred_boxes, None], [target_boxes, None])
        total_loss = losses[0]

        for d in losses[1:]:
            loss_dict.update(d)

        for d in metrics:
            metric_dict.update(d)

        return total_loss, [loss_dict, metric_dict]

    def track(self, images: Tensor, templates: Tensor, **kwargs):

        ns, _, hs, ws = images.shape

        # ----------- backbone feature -------------------------
        s_feat, t_feat = self.get_backbone_feature(images, templates)

        # ----------- fusion -------------------------
        feat = self.neck(s_feat, t_feat)

        # ----------- predict -------------------------
        pred_boxes, score_lt, score_br = self.head(feat)

        # ----------- convert -------------------------
        outputs_coord = box_convert(pred_boxes, in_fmt='xyxy', out_fmt='cxcywh')

        pred_dict = dict()
        pred_dict['box'] = outputs_coord.squeeze().detach().cpu().numpy()
        pred_dict['score'] = score_lt.max().item() * score_br.max().item()
        pred_dict['visualize'] = [None, None]

        return pred_dict

    def get_backbone_feature(self, x, z):
        if self.cfg.backbone.type == 'ResNet':
            x = self.backbone(self._imagenet_norm(x))[-1]
            z = self.backbone(self._imagenet_norm(z))[-1]
            x = x.flatten(2).permute(0, 2, 1)
            z = z.flatten(2).permute(0, 2, 1)
        else:
            x, z = self.backbone(self._imagenet_norm(x), self._imagenet_norm(z))
        return x, z


def build_translate_track(args):
    model = Model(args)
    return model


if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    from config.cfg_translation_track import cfg

    net = build_translate_track(cfg.model)

    gt = torch.Tensor([[0.1, 0.3, 0.7, 0.8], [0.4, 0.3, 0.7, 0.5]])
    x = torch.rand(2, 3, cfg.model.search_size[0], cfg.model.search_size[1])
    z = torch.rand(2, 3, cfg.model.template_size[0], cfg.model.template_size[1])

    in_dict = {
        'search': x,
        'template': z,
        'target': gt,
        'training': True,
    }

    out = net(in_dict)
    print(out)


    def prepare_input(resolution):
        input_dict = {
            'search': x,
            'template': z,
            'target': gt,
            'training': True,
        }

        return dict(input_dict=input_dict)


    flops, params = get_model_complexity_info(net,
                                              input_res=(None,),
                                              input_constructor=prepare_input,
                                              as_strings=True, print_per_layer_stat=False)
    print('      - Macs:  ' + flops)
    print('      - Params: ' + params)
