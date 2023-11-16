import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class Tracker(object):
    def __init__(self):
        super(Tracker, self).__init__()
        self.set_deterministic()

        # causes cuDNN to deterministically select an algorithm, possibly at the cost of reduced performance
        torch.backends.cudnn.benchmark = False

        # While disabling CUDA convolution benchmarking (discussed above) ensures that CUDA selects the same algorithm
        # each time an application is run, that algorithm itself may be nondeterministic, unless either
        # torch.use_deterministic_algorithms(True) or torch.backends.cudnn.deterministic = True is set.
        # The latter setting controls only this behavior, unlike torch.use_deterministic_algorithms()
        # which will make other PyTorch operations behave deterministically, too.
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"  # <-- for cuda>=10.2
        if int(torch.__version__.split('.')[0]) == 1 and int(torch.__version__.split('.')[1]) >= 8:  # pytorch>=1.8
            torch.use_deterministic_algorithms(True)
        else:
            torch.backends.cudnn.deterministic = True

        # updated hyper-params
        self.vis = False
        self.fp16 = False

        self.template_sf = None
        self.template_sz = None

        self.search_sf = None
        self.search_sz = None

        # ---------------
        self.model = None

        self.template_feat_sz = None
        self.search_feat_sz = None

        self.template_info = None

        self.language = None  # (B, 768)
        self.init_box = None  # [x y x y]
        self.last_box = None
        self.last_pos = None
        self.last_size = None
        self.last_score = None
        self.last_image = None

        self.imw = None
        self.imh = None
        self.channel_average = None

        self.idx = 0

    @staticmethod
    def set_deterministic():
        torch.cuda.manual_seed(123)
        torch.manual_seed(123)
        np.random.seed(123)
        random.seed(123)

    def init(self, *args, **kwargs):
        raise NotImplementedError

    def track(self, *args, **kwargs):
        raise NotImplementedError

    def update_state(self, pred_box, pred_score, scale_f):  # [cx cy w h]
        sf_w, sf_h, _, _ = scale_f

        if pred_score is None:
            self.last_score = 1
        else:
            self.last_score = pred_score

        delta_pos = (pred_box[:2] - 0.5) * self.search_sz / np.array([sf_w, sf_h])
        current_size = pred_box[2:] * self.search_sz / np.array([sf_w, sf_h])

        self.last_pos = self.last_pos + delta_pos
        self.last_size = current_size

        self.last_box = np.array([self.last_pos[0] - self.last_size[0] / 2,
                                  self.last_pos[1] - self.last_size[1] / 2,
                                  self.last_pos[0] + self.last_size[0] / 2,
                                  self.last_pos[1] + self.last_size[1] / 2])

        out_box = np.array(self.last_box)
        out_box = self.clip_box(out_box, margin=10)

        return out_box, self.last_score  # [x y w h]

    def update_hyper_params(self, hp: dict):
        if hp is not None:
            for key, value in hp.items():
                setattr(self, key, value)

    @staticmethod
    def crop_patch(im, box, scale_factor, out_size, mean_value):  # [x, y, x, y]
        pos = (box[:2] + box[2:]) / 2
        wh = box[2:] - box[:2] + 1

        w_z = wh[0] + (scale_factor - 1) * np.mean(wh)
        h_z = wh[1] + (scale_factor - 1) * np.mean(wh)
        crop_sz = np.ceil(np.sqrt(w_z * h_z))

        x1 = pos[0] - crop_sz / 2
        y1 = pos[1] - crop_sz / 2

        a = out_size / crop_sz
        b = out_size / crop_sz
        c = -a * x1
        d = -b * y1

        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)

        patch = cv2.warpAffine(im, mapping,
                               (out_size, out_size),
                               borderMode=cv2.BORDER_CONSTANT,
                               # borderValue=np.array([123.675, 116.28, 103.53]))  # RGB
                               borderValue=np.mean(im, axis=(0, 1)) if mean_value is None else mean_value)

        x1, y1, x2, y2 = box
        out_box = np.array([x1, y1, x2, y2])

        out_box[0::2] = out_box[0::2] * a + c
        out_box[1::2] = out_box[1::2] * b + d

        out_box[0::2] = np.clip(out_box[0::2], 0, out_size - 1)
        out_box[1::2] = np.clip(out_box[1::2], 0, out_size - 1)  # [x, y, x, y]

        return patch, out_box, [a, b, c, d]

    @staticmethod
    def crop_patch_fast(im, box, scale_factor, out_size):  # [x, y, x, y]
        pos = (box[:2] + box[2:]) / 2
        wh = box[2:] - box[:2] + 1

        w_z = wh[0] + (scale_factor - 1) * np.mean(wh)
        h_z = wh[1] + (scale_factor - 1) * np.mean(wh)
        crop_sz = np.ceil(np.sqrt(w_z * h_z))

        x1 = pos[0] - crop_sz / 2
        y1 = pos[1] - crop_sz / 2

        a = out_size / crop_sz
        b = out_size / crop_sz
        c = -a * x1
        d = -b * y1

        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)

        patch = cv2.warpAffine(im, mapping,
                               (out_size, out_size),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=np.array([123.675, 116.28, 103.53]))  # RGB

        x1, y1, x2, y2 = box
        out_box = np.array([x1, y1, x2, y2])

        out_box[0::2] = out_box[0::2] * a + c
        out_box[1::2] = out_box[1::2] * b + d

        out_box[0::2] = np.clip(out_box[0::2], 0, out_size - 1)
        out_box[1::2] = np.clip(out_box[1::2], 0, out_size - 1)  # [x, y, x, y]

        return patch, out_box, [a, b, c, d]

    # pysot version -- get_subwindow
    @staticmethod
    def crop_patch_pysot(im, box, scale_factor, out_size):  # [x, y, x, y]

        pos = (box[:2] + box[2:]) / 2
        wh = box[2:] - box[:2] + 1

        w_z = wh[0] + (scale_factor - 1) * np.mean(wh)
        h_z = wh[1] + (scale_factor - 1) * np.mean(wh)
        crop_sz = np.ceil(np.sqrt(w_z * h_z))

        x1 = pos[0] - crop_sz / 2
        y1 = pos[1] - crop_sz / 2

        a = out_size / crop_sz
        b = out_size / crop_sz
        c = -a * x1
        d = -b * y1

        x1, y1, x2, y2 = box
        out_box = np.array([x1, y1, x2, y2])

        out_box[0::2] = out_box[0::2] * a + c
        out_box[1::2] = out_box[1::2] * b + d

        out_box[0::2] = np.clip(out_box[0::2], 0, out_size - 1)
        out_box[1::2] = np.clip(out_box[1::2], 0, out_size - 1)  # [x, y, x, y]

        # ----------------------
        original_sz = crop_sz
        sz = original_sz
        im_sz = im.shape
        model_sz = out_size
        avg_chans = np.mean(im, axis=(0, 1))

        # -------------------------------------------------

        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))

        # im_patch = im_patch.transpose(2, 0, 1)
        # im_patch = im_patch[np.newaxis, :, :, :]
        # im_patch = im_patch.astype(np.float32)
        # im_patch = torch.from_numpy(im_patch)

        return im_patch, out_box, [a, b]

    @staticmethod
    def crop_patch_pytracking(im, box, scale_factor, out_size,
                              # pos: torch.Tensor, sample_sz: torch.Tensor,
                              # output_sz: torch.Tensor = None,
                              mode: str = 'replicate', max_scale_change=None, is_mask=False):
        """Sample an image patch.     def crop_patch(im, box, scale_factor, out_size):  # [x, y, x, y]

        args:
            im: Image
            pos: center position of crop
            sample_sz: size to crop
            output_sz: size to resize to
            mode: how to treat image borders: 'replicate' (default), 'inside' or 'inside_major'
            max_scale_change: maximum allowed scale change when using 'inside' and 'inside_major' mode
        """

        # if mode not in ['replicate', 'inside']:
        #     raise ValueError('Unknown border mode \'{}\'.'.format(mode))

        pos = (box[:2] + box[2:]) / 2
        wh = box[2:] - box[:2] + 1

        w_z = wh[0] + (scale_factor - 1) * np.mean(wh)
        h_z = wh[1] + (scale_factor - 1) * np.mean(wh)
        crop_sz = np.ceil(np.sqrt(w_z * h_z))

        # -----------------------------
        im = torch.from_numpy(im).float().permute(2, 0, 1).unsqueeze(0)
        sample_sz = torch.tensor(crop_sz).float()
        output_sz = torch.tensor(out_size).float()
        pos = torch.tensor([pos[1], pos[0]]).float()

        # copy and convert
        posl = pos.long().clone()

        pad_mode = mode

        # # Get new sample size if forced inside the image
        # if mode == 'inside' or mode == 'inside_major':
        #     pad_mode = 'replicate'
        #     im_sz = torch.Tensor([im.shape[2], im.shape[3]])
        #     shrink_factor = (sample_sz.float() / im_sz)
        #     if mode == 'inside':
        #         shrink_factor = shrink_factor.max()
        #     elif mode == 'inside_major':
        #         shrink_factor = shrink_factor.min()
        #     shrink_factor.clamp_(min=1, max=max_scale_change)
        #     sample_sz = (sample_sz.float() / shrink_factor).long()

        # Compute pre-downsampling factor
        resize_factor = torch.min(sample_sz.float() / output_sz.float()).item()
        df = int(max(int(resize_factor - 0.1), 1))

        sz = sample_sz.float() / df  # new size
        wh = wh / df

        # Do downsampling
        if df > 1:
            os = posl % df  # offset
            posl = (posl - os) // df  # new position
            im2 = im[..., os[0].item()::df, os[1].item()::df]  # downsample
        else:
            im2 = im

        # compute size to crop
        szl = torch.max(sz.round(), torch.Tensor([2])).long()

        # Extract top and bottom coordinates
        tl = posl - (szl - 1) // 2
        br = posl + szl // 2 + 1

        # # Shift the crop to inside
        # if mode == 'inside' or mode == 'inside_major':
        #     im2_sz = torch.LongTensor([im2.shape[2], im2.shape[3]])
        #     shift = (-tl).clamp(0) - (br - im2_sz).clamp(0)
        #     tl += shift
        #     br += shift
        #
        #     outside = ((-tl).clamp(0) + (br - im2_sz).clamp(0)) // 2
        #     shift = (-tl - outside) * (outside > 0).long()
        #     tl += shift
        #     br += shift
        #
        #     # Get image patch
        #     # im_patch = im2[...,tl[0].item():br[0].item(),tl[1].item():br[1].item()]

        # Get image patch
        if not is_mask:
            im_patch = F.pad(im2,
                             (-tl[1].item(), br[1].item() - im2.shape[3], -tl[0].item(), br[0].item() - im2.shape[2]),
                             pad_mode)
        else:
            im_patch = F.pad(im2,
                             (-tl[1].item(), br[1].item() - im2.shape[3], -tl[0].item(), br[0].item() - im2.shape[2]))

        # Get image coordinates
        # patch_coord = df * torch.cat((tl, br)).view(1, 4)
        tmp_sx = output_sz / im_patch.shape[2]
        tmp_sy = output_sz / im_patch.shape[3]
        pos_x = posl[1] - tl[1].item()
        pos_y = posl[0] - tl[0].item()

        out_box = np.array([(pos_x - wh[0] / 2) * tmp_sx,
                            (pos_y - wh[1] / 2) * tmp_sy,
                            (pos_x + wh[0] / 2 - 1) * tmp_sx,
                            (pos_y + wh[1] / 2 - 1) * tmp_sy
                            ])

        # Resample
        if not is_mask:
            im_patch = F.interpolate(im_patch, output_sz.long().tolist(), mode='bilinear')
        else:
            im_patch = F.interpolate(im_patch, output_sz.long().tolist(), mode='nearest')

        return im_patch, out_box, [1/df, 1/df, -1, -1]

    @staticmethod
    def to_pytorch(x):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x.transpose(2, 0, 1)).cuda().unsqueeze(0)
        else:
            x = x.cuda()

        return x

    def clip_box(self, box, margin=0):
        imh = self.imh
        imw = self.imw

        x1, y1, x2, y2 = box

        x1 = min(max(0, x1), imw - margin)
        x2 = min(max(margin, x2), imw)
        y1 = min(max(0, y1), imh - margin)
        y2 = min(max(margin, y2), imh)
        w = max(margin, x2 - x1)
        h = max(margin, y2 - y1)

        if x1 == 0 and y1 == 0 and x2 == imw and y2 == imh:
            x1, y1, x2, y2 = self.init_box
            w = x2 - x1
            h = y2 - y1
            x1, y1, x2, y2 = np.array([self.imw / 2 - w / 2, self.imh / 2 - h / 2,
                                       self.imw / 2 + w / 2, self.imh / 2 + h / 2])
            self.last_box = np.array([x1, y1, x2, y2])

            w = max(margin, x2 - x1)
            h = max(margin, y2 - y1)

        return np.array([x1, y1, w, h])
