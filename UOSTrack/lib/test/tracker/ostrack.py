import math
from copy import deepcopy

from UOSTrack.lib.models.ostrack import build_ostrack
from UOSTrack.lib.test.tracker.basetracker import BaseTracker
import torch
import numpy as np
import time
from UOSTrack.lib.test.utils.hann import hann2d
from UOSTrack.lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from UOSTrack.lib.test.tracker.data_utils import Preprocessor
from UOSTrack.lib.utils.box_ops import clip_box
from UOSTrack.lib.utils.ce_utils import generate_mask_cond

from UOSTrack.lib.test.tracker.kftool import KalmanBoxTracker
from UOSTrack.lib.test.tracker.kf_utils import decode_muli_bbox
from UOSTrack.lib.utils.box_ops import box_iou
from UOSTrack.lib.test.tracker.kf_utils import deep_xywh2xyxy, NMS

from UOSTrack.external.uie.FUnIE_GAN import build_fuinegan
from UOSTrack.external.uie.RGHS import RGHSUWE
from UOSTrack.external.uie.UCM import UCM
from UOSTrack.external.uie.Shallow_UWnet import build_shallowuwnet
from UOSTrack.external.uie.Ushape import build_ushape


class OSTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(OSTrack, self).__init__(params)

        network = build_ostrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE

        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)

        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}

        # using kalman filter to head
        self.use_kf = True
        self.kalmanFilter = None
        self.num = 30
        self.nms_thre = 0.8

        # using uie
        self.use_uie = False
        self.uie = build_fuinegan()

        #print('use kf', self.use_kf, 'use uie', self.use_uie)

    def initialize(self, image, info: dict):
        H, W, _ = image.shape

        if self.use_uie:
            image = self.uie(image)

        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                                output_sz=self.params.template_size)

        # raise
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)

        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None

        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)

            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0

        if self.use_kf:
            kf_init_box = [info['init_bbox'][0], info['init_bbox'][1], info['init_bbox'][0] + info['init_bbox'][2],
                           info['init_bbox'][1] + info['init_bbox'][3]]
            self.kalmanFilter = KalmanBoxTracker(kf_init_box)

        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        if self.use_uie:
            image = self.uie(image)

        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)

        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=self.z_dict1.tensors, search=x_dict.tensors, ce_template_mask=self.box_mask_z)

        #  origin box
        response_origin = self.output_window * out_dict['score_map']
        pred_boxes_origin = self.network.box_head.cal_bbox(response_origin, out_dict['size_map'],
                                                           out_dict['offset_map'])

        pred_boxes_origin = pred_boxes_origin.view(-1, 4)
        pred_box_origin = (pred_boxes_origin.mean(dim=0) * self.params.search_size / resize_factor).tolist()

        # get the final box result
        max_response_box = clip_box(self.map_box_back(pred_box_origin, resize_factor), H, W, margin=10)

        #  use kf
        if self.use_kf:
            response = out_dict['score_map']
            size_map = out_dict['size_map']
            offset_map = out_dict['offset_map']

            pred_box_list, response_score = decode_muli_bbox(response, size_map, offset_map, self.num, self.feat_sz,
                                                             self.state, resize_factor, H, W)
            pred_box_list, response_score = NMS(pred_box_list, response_score, iou_threshold=self.nms_thre)
            kf_pred_bbox = self.kalmanFilter.predict()

            iou_result, _ = box_iou(deep_xywh2xyxy(torch.tensor(pred_box_list)), torch.tensor(kf_pred_bbox))

            iou_result = torch.mul(iou_result, torch.tensor(response_score))

            if box_iou(deep_xywh2xyxy(torch.tensor(max_response_box).unsqueeze(dim=0)), torch.tensor(kf_pred_bbox))[
                0] > 0.6:
                self.state = max_response_box
            else:
                index = torch.where(iou_result == torch.sort(iou_result)[0][- 1])[0]
                self.state = list(map(float, pred_box_list[int(index.tolist()[0])]))

        else:
            self.state = max_response_box

        if self.use_kf:
            #  update kalman filter
            kf_update_bbox = [self.state[0], self.state[1], self.state[0] + self.state[2],
                              self.state[1] + self.state[3]]
            self.kalmanFilter.update(kf_update_bbox)

        return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return OSTrack
