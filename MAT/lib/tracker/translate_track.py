import cv2
import os
import torch
import numpy as np
from copy import deepcopy

from ._tracker import Tracker


class TranslateT(Tracker):
    def __init__(self, hyper: dict, model):
        super(TranslateT, self).__init__()

        # updated hyper-params
        self.vis = False

        self.template_sf = None
        self.template_sz = None

        self.search_sf = None
        self.search_sz = None

        # --------------- hyper of this tracker
        self.score_threshold = None

        self.update_hyper_params(hyper)

        self.template_sz = self.template_sz[0]
        self.search_sz = self.search_sz[0]
        # ---------------
        self.model = model

        self.template_feat_sz = self.template_sz // self.model.backbone.out_stride
        self.search_feat_sz = self.search_sz // self.model.backbone.out_stride

        self.template_info = None

        self.language = None  # (N, L, 768)
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

    def init(self, im, gt, **kwargs):  # BGRimg [x y w h]
        if self.vis:
            cv2.namedWindow('CommonTracker', cv2.WINDOW_NORMAL)

        self.set_deterministic()

        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        x, y, w, h = gt

        self.idx = 1

        self.imh, self.imw = im.shape[:2]
        self.channel_average = np.mean(im, axis=(0, 1))

        self.init_box = np.array([x, y, x+w-1, y+h-1])
        self.last_box = np.array([x, y, x+w-1, y+h-1])
        self.last_score = 1
        self.last_image = np.array(im)

        self.last_pos = np.array([x+w/2, y+h/2])
        self.last_size = np.array([w, h])

        template_patch, template_roi, scale_f = self.crop_patch_fast(
            self.last_image, self.init_box, scale_factor=self.template_sf, out_size=self.template_sz,
        )
        self.template_info = self.to_pytorch(template_patch)

    def track(self, im, **kwargs):
        curr_image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        self.idx += 1

        curr_patch, last_roi, scale_f = self.crop_patch_fast(
            curr_image, self.last_box, scale_factor=self.search_sf, out_size=self.search_sz,
        )

        with torch.no_grad():
            pred_dict = self.model.track(self.to_pytorch(curr_patch), self.template_info)

        pred_box = pred_dict['box']
        pred_score = pred_dict['score']

        out_box, out_score = self.update_state(pred_box, pred_score, scale_f)

        if self.vis:
            bb = np.array(out_box).astype(int)
            bb[2:] = bb[2:] + bb[:2] - 1
            im = cv2.rectangle(im, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 4)
            cv2.putText(im, '{:.2f}'.format(out_score), (40, 60), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
            cv2.imshow('CommonTracker', im)
            cv2.waitKey(1)

        return out_box, out_score, pred_dict['visualize']  # [x y w h]

