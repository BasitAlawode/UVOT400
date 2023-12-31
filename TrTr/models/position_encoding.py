"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

from TrTr.util.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.segment_embdded_factor = 0.0  #0.5, 0.0 gives best result for VOT2018. TODO: hyperparameter

    def forward(self, tensor_list: NestedTensor, multi_frame = False):
        x = tensor_list.tensors

        #print("x: {}".format(x.shape))

        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        #not_mask = torch.ones(tensor_list.mask.shape, device = tensor_list.mask.device) # Adding position embedding to maks area (average padding area) will degrade the performance
        # Note: can not use different model between training and inference
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        #print("x_embed: {}".format(x_embed.shape))
        #print("x_embed[:, :, -1:]: {}".format(x_embed[:, :, -1:].shape))
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        #print("x_embed: {}".format(x_embed.shape))
        #print("y_embed: {}".format(y_embed.shape))
        #print("dim_t: {}".format(dim_t))

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        #print("pos_x: {}".format(pos_x.shape))
        #print("pos_y: {}".format(pos_y.shape))


        #print("stack: {}".format(torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).shape))
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        #print("pos_x: {}".format(pos_x.shape))
        #print("pos_y: {}".format(pos_y.shape))

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        #print("pos: {}".format(pos.shape))

        # add an additioanl segment (frame) embedding for multilple frames.
        if multi_frame:
            # a temporal segment embedding for two frames => TODO: should learn?
            assert(len(pos) == 2)
            pos[-1].add_(self.segment_embdded_factor)

        return pos

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(args):
    N_steps = args.transformer.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
