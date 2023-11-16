import pprint

import torch
import torch.nn as nn

from lib.model.backbones.MAE import mae_vit_base_patch16, mae_vit_small_patch16


class MAE(nn.Module):

    def __init__(self,
                 arch: str,
                 train_flag: bool = False,
                 train_all: bool = False,
                 weights: str = None,
                 from_scratch: bool = False,
                 ):
        super(MAE, self).__init__()

        if 'base' in arch:
            self.model = mae_vit_base_patch16()

            if not from_scratch:
                ckp_dict = torch.load('/home/space/Documents/Experiments/BaseT/pretrain/mae_pretrain_vit_base_full.pth',
                                      map_location='cpu')['model']

                model_dict = self.model.state_dict()

                pretrained_dict = {k: v for k, v in ckp_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
                unused_param = [k for k, v in ckp_dict.items() if k not in model_dict]
                lost_param = [k for k, v in model_dict.items() if k not in ckp_dict or v.shape != ckp_dict[k].shape]

                model_dict.update(pretrained_dict)
                self.model.load_state_dict(model_dict)

                print('unused param:')
                pprint.pprint(sorted(unused_param))
                print('lost_param:')
                pprint.pprint(sorted(lost_param))

                del ckp_dict

                if weights is not None:
                    print('load pretrain encoder from:', weights.split('/')[-1])
                    ckp_dict = torch.load(weights, map_location='cpu')['model']
                    ckp_dict = {k.replace('backbone.model.', ''): v for k, v in ckp_dict.items()}
                    model_dict = self.model.state_dict()

                    pretrained_dict = {k: v for k, v in ckp_dict.items() if
                                       k in model_dict and v.shape == model_dict[k].shape}
                    unused_param = [k for k, v in ckp_dict.items() if k not in model_dict]
                    lost_param = [k for k, v in model_dict.items() if k not in ckp_dict or v.shape != ckp_dict[k].shape]

                    model_dict.update(pretrained_dict)
                    self.model.load_state_dict(model_dict)

                    print('unused param:')
                    pprint.pprint(sorted(unused_param))
                    print('lost_param:')
                    pprint.pprint(sorted(lost_param))

            else:
                print('train from scratch')
        elif 'small' in arch:
            self.model = mae_vit_small_patch16()

            if not from_scratch:
                ckp_dict = torch.load('/home/space/Documents/Experiments/BaseT/pretrain/mae_pretrain_vit_base_full.pth',
                                      map_location='cpu')['model']

                model_dict = self.model.state_dict()

                pretrained_dict = {k: v for k, v in ckp_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
                unused_param = [k for k, v in ckp_dict.items() if k not in model_dict]
                lost_param = [k for k, v in model_dict.items() if k not in ckp_dict or v.shape != ckp_dict[k].shape]

                model_dict.update(pretrained_dict)
                self.model.load_state_dict(model_dict)

                print('unused param:')
                pprint.pprint(sorted(unused_param))
                print('lost_param:')
                pprint.pprint(sorted(lost_param))

                del ckp_dict

                if weights is not None:
                    print('load pretrain encoder from:', weights.split('/')[-1])
                    ckp_dict = torch.load(weights, map_location='cpu')['model']
                    ckp_dict = {k.replace('backbone.model.', ''): v for k, v in ckp_dict.items()}
                    model_dict = self.model.state_dict()

                    pretrained_dict = {k: v for k, v in ckp_dict.items() if
                                       k in model_dict and v.shape == model_dict[k].shape}
                    unused_param = [k for k, v in ckp_dict.items() if k not in model_dict]
                    lost_param = [k for k, v in model_dict.items() if k not in ckp_dict or v.shape != ckp_dict[k].shape]

                    model_dict.update(pretrained_dict)
                    self.model.load_state_dict(model_dict)

                    print('unused param:')
                    pprint.pprint(sorted(unused_param))
                    print('lost_param:')
                    pprint.pprint(sorted(lost_param))
            else:
                print('train from scratch')
        else:
            raise NotImplementedError

        if train_flag:
            for name, parameter in self.model.named_parameters():
                parameter.requires_grad_(False)

            if train_all:
                for name, parameter in self.model.named_parameters():
                    if 'pos_embed' not in name:  # fixed sin-cos embedding
                        parameter.requires_grad_(True)
            else:
                for name, parameter in self.model.blocks.named_parameters():
                    parameter.requires_grad_(True)
                for name, parameter in self.model.norm.named_parameters():
                    parameter.requires_grad_(True)

        else:
            for name, parameter in self.model.named_parameters():
                parameter.requires_grad_(False)

    def forward(self, x, z_in, z_out, mask_ratio=0.25):
        outputs = self.model(x, z_in, z_out, mask_ratio=mask_ratio)
        if len(outputs) > 2:
            loss, y, mask, loss_s, s = outputs
            y = self.model.unpatchify(y)
            s = self.model.unpatchify(s)

            return loss, y, mask, loss_s, s
        else:
            loss_s, s = outputs
            s = self.model.unpatchify(s)
            return loss_s, s


def build_backbone(_args):
    model = MAE(arch=_args.arch, train_flag=_args.lr_mult > 0, train_all=_args.train_all,
                weights=_args.weights, from_scratch=_args.from_scratch)

    return model


if __name__ == '__main__':
    from config.cfg_translation import cfg

    backbone = build_backbone(cfg.model.backbone)

    x = torch.rand(1, 3, 224, 224)
    z = torch.rand(1, 3, 112, 112)
    ys = backbone(x, z, z)
    print([_y.shape for _y in ys])
    from ptflops import get_model_complexity_info

    def prepare_input(resolution):

        input_dict = {
            'x': x,
            'z_in': z,
            'z_out': z,
        }

        return input_dict
    flops, params = get_model_complexity_info(backbone,
                                              input_res=(x.shape,),
                                              input_constructor=prepare_input,
                                              as_strings=True, print_per_layer_stat=False)
    print('      - Flops:  ' + flops)
    print('      - Params: ' + params)
    #       - Flops:  16.95 GMac -- base
    #       - Params: 112.25 M
    #       - Flops:  62.17 GMac -- large
    #       - Params: 330.03 M
