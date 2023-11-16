import pprint
from copy import deepcopy

import torch
import torch.nn as nn

from MAT.lib.model.backbones.MAE import mae_vit_base_patch16, mae_vit_small_patch16


class MAEEncode(nn.Module):

    def __init__(self,
                 arch: str,
                 train_flag: bool = False,
                 train_all: bool = False,
                 weights: str = None,
                 train_layers: list = []
                 ):
        super(MAEEncode, self).__init__()

        if 'base' in arch:
            model = mae_vit_base_patch16()

            if weights is not None:
                print('load pretrain encoder from:', weights.split('/')[-1])
                ckp_dict = torch.load(weights, map_location='cpu')['model']
                ckp_dict = {k.replace('backbone.model.', ''): v for k, v in ckp_dict.items()}
                model_dict = model.state_dict()

                pretrained_dict = {k: v for k, v in ckp_dict.items() if
                                   k in model_dict and v.shape == model_dict[k].shape}
                unused_param = [k for k, v in ckp_dict.items() if k not in model_dict]
                lost_param = [k for k, v in model_dict.items() if k not in ckp_dict or v.shape != ckp_dict[k].shape]
                print('unused param:')
                pprint.pprint(sorted(unused_param))
                print('lost_param:')
                pprint.pprint(sorted(lost_param))

                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
            else:
                ckp_dict = torch.load('/home/space/Documents/Experiments/BaseT/pretrain/mae_pretrain_vit_base_full.pth',
                                      map_location='cpu')['model']

                model_dict = model.state_dict()

                pretrained_dict = {k: v for k, v in ckp_dict.items() if
                                   k in model_dict and v.shape == model_dict[k].shape}
                unused_param = [k for k, v in ckp_dict.items() if k not in model_dict]
                lost_param = [k for k, v in model_dict.items() if k not in ckp_dict or v.shape != ckp_dict[k].shape]
                print('unused param:')
                pprint.pprint(sorted(unused_param))
                print('lost_param:')
                pprint.pprint(sorted(lost_param))

                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)

            self.patch_embed = deepcopy(model.patch_embed)
            self.pos_embed = deepcopy(model.pos_embed)  # fixed sin-cos embedding

            self.z_patch_embed = deepcopy(model.z_patch_embed)
            self.z_pos_embed = deepcopy(model.z_pos_embed)  # fixed sin-cos embedding

            self.cls_token = deepcopy(model.cls_token)
            self.blocks = deepcopy(model.blocks)
            self.norm = deepcopy(model.norm)

            self.out_stride = self.patch_embed.patch_size[0]
            self.out_channels_list = [self.cls_token.shape[-1]]

            del model
            del ckp_dict

        elif 'small' in arch:
            model = mae_vit_small_patch16()

            if weights is not None:
                print('load pretrain encoder from:', weights.split('/')[-1])
                ckp_dict = torch.load(weights, map_location='cpu')['model']
                ckp_dict = {k.replace('backbone.model.', ''): v for k, v in ckp_dict.items()}
                model_dict = model.state_dict()

                pretrained_dict = {k: v for k, v in ckp_dict.items() if
                                   k in model_dict and v.shape == model_dict[k].shape}
                unused_param = [k for k, v in ckp_dict.items() if k not in model_dict]
                lost_param = [k for k, v in model_dict.items() if k not in ckp_dict or v.shape != ckp_dict[k].shape]
                print('unused param:')
                pprint.pprint(sorted(unused_param))
                print('lost_param:')
                pprint.pprint(sorted(lost_param))

                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
            else:
                raise NotImplementedError  # mae does not have small vit

            self.patch_embed = deepcopy(model.patch_embed)
            self.pos_embed = deepcopy(model.pos_embed)  # fixed sin-cos embedding

            self.z_patch_embed = deepcopy(model.z_patch_embed)
            self.z_pos_embed = deepcopy(model.z_pos_embed)  # fixed sin-cos embedding

            self.cls_token = deepcopy(model.cls_token)
            self.blocks = deepcopy(model.blocks)
            self.norm = deepcopy(model.norm)

            self.out_stride = self.patch_embed.patch_size[0]
            self.out_channels_list = [self.cls_token.shape[-1]]

            del model
            del ckp_dict
        else:
            raise NotImplementedError

        if train_flag:
            for name, parameter in self.named_parameters():
                parameter.requires_grad_(False)

            if train_all:
                for name, parameter in self.named_parameters():
                    if 'pos_embed' not in name:  # fixed sin-cos embedding
                        parameter.requires_grad_(True)
            else:
                for id in train_layers:
                    for name, parameter in self.blocks[id].named_parameters():
                        parameter.requires_grad_(True)
                for name, parameter in self.norm.named_parameters():
                    parameter.requires_grad_(True)

        else:
            for name, parameter in self.named_parameters():
                parameter.requires_grad_(False)

    def forward(self, x, z):
        # embed patches
        x = self.patch_embed(x)
        z = self.z_patch_embed(z)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        z = z + self.z_pos_embed[:, 0:, :]

        len_z = z.shape[1]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, z, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        z = x[:, 1:1 + len_z]
        x = x[:, 1 + len_z:]
        return x, z


def build_backbone(_args):
    model = MAEEncode(arch=_args.arch, train_flag=_args.lr_mult > 0, train_all=_args.train_all,
                      weights=_args.weights, train_layers=_args.train_layers)

    return model


if __name__ == '__main__':
    from config.cfg_translation_track import cfg as exp

    backbone = build_backbone(exp.model.backbone)

    x = torch.rand(1, 3, 224, 224)
    z = torch.rand(1, 3, 112, 112)
    ys = backbone(x, z)
    print([_y.shape for _y in ys])

    from ptflops import get_model_complexity_info


    def prepare_input(resolution):
        input_dict = {
            'x': x,
            'z': z,
        }

        return input_dict


    flops, params = get_model_complexity_info(backbone,
                                              input_res=(None,),
                                              input_constructor=prepare_input,
                                              as_strings=True, print_per_layer_stat=False)
    print('      - Flops:  ' + flops)
    print('      - Params: ' + params)
    #       - Flops:  21.04 GMac
    #       - Params: 42.53 M
