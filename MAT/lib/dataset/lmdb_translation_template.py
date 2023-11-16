import cv2
import random
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import lmdb
import albumentations as aug

import torch
from lib.dataset._dataset import SubSet, BaseDataset

"""
Examples:

tmp = {
    'data_set': data_set,
    'name'
    'num'
    'length'
}
data_set = [
    [
        {'path':[images/airplane-10/img/00000001.jpg], 
         'bbox':[x y w h], 
         'size':[im_w, im_h], 
         'name': lasot, ...
         }, 
         ...
    ],
    ...
]
"""


def lmdb_translation_template_collate_fn(batch):
    template_in = [torch.Tensor(item[0]).unsqueeze(0) for item in batch]
    search_img = [torch.Tensor(item[1]).unsqueeze(0) for item in batch]
    template_out = [torch.Tensor(item[2]).unsqueeze(0) for item in batch]

    template_in = torch.cat(template_in, dim=0)
    search_img = torch.cat(search_img, dim=0)
    template_out = torch.cat(template_out, dim=0)

    return {
        'template_in': template_in,
        'search': search_img,
        'template_out': template_out,
    }


class LMDBPatch(BaseDataset):

    def __init__(self, cfg, lmdb_path, json_path, dataset_name_list: list = None, num_samples: int = None):
        super(LMDBPatch).__init__()

        self.debug = False
        if self.debug:
            self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3)

        self.lmdb_path: dict = lmdb_path
        self.json_path: dict = json_path

        self.sample_range: int = cfg.sample_range

        self.search_sz: List[int, int] = cfg.search_size
        self.search_scale_f: float = cfg.search_scale_f
        self.search_jitter_f: List[float, float] = cfg.search_jitter_f

        self.template_sz: List[int, int] = cfg.template_size
        self.template_scale_f: float = cfg.template_scale_f
        self.template_jitter_f: List[float, float] = cfg.template_jitter_f

        # Declare an augmentation pipeline
        self.aug = aug.Compose([
            aug.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, p=1),
        ])

        self.in_aug = aug.Compose([
            # aug.CoarseDropout(always_apply=False, p=1.0,
            #                   max_holes=15, max_height=10, max_width=10, min_holes=8, min_height=8, min_width=8,
            #                   fill_value=[123.675, 116.28, 103.53]),
            aug.MotionBlur(always_apply=False, p=0.2, blur_limit=(3, 15)),
        ])

        # load dataset
        self.LMDB_ENVS = {}
        self.LMDB_HANDLES = {}
        self.video_list: List = []
        for name in dataset_name_list:
            if isinstance(name, list):
                use_num = int(name[1])
                name = name[0]

                dataset = SubSet(name=name, load=self.json_path[name])
                random.shuffle(dataset.data_set)
                self.video_list += dataset.data_set[:use_num]
            else:
                dataset = SubSet(name=name, load=self.json_path[name])
                self.video_list += dataset.data_set

            env = lmdb.open(self.lmdb_path[name], readonly=True, lock=False, readahead=False, meminit=False)
            self.LMDB_ENVS[name] = env
            item = env.begin(write=False)
            self.LMDB_HANDLES[name] = item

        # repeat and shuffle
        if len(dataset_name_list) > 1:
            random.shuffle(self.video_list)
        while len(self.video_list) < num_samples:
            self.video_list += self.video_list
        self.video_list = self.video_list[:num_samples]
        self.video_num = len(self.video_list)
        random.shuffle(self.video_list)

    def __len__(self):
        return self.video_num

    def __getitem__(self, item):
        t_dict, s_dict = self.check_sample(self.video_list[item], self.video_list, self.sample_range)

        # read RGB image, [x y w h]
        template_img, t_box, t_lang = self.parse_frame_lmdb(t_dict, self.LMDB_HANDLES, need_language=True)
        search_img, s_box, s_lang = self.parse_frame_lmdb(s_dict, self.LMDB_HANDLES, need_language=True)

        if isinstance(s_lang, list):
            s_lang = random.choice(s_lang)

        template_in, _, t_box = self.crop_square_fast(
            template_img, t_box,
            out_size=self.template_sz,
            scale_factor=self.template_scale_f,
            jitter_f=self.template_jitter_f)

        search_img, _, s_box = self.crop_patch_fast(
            search_img, s_box,
            out_size=self.search_sz,
            scale_factor=self.search_scale_f,
            jitter_f=self.search_jitter_f)

        template_in, search_img = map(lambda im: self.aug(image=im)["image"], [template_in, search_img])
        template_in = self.in_aug(image=template_in)["image"]  # motion blur

        template_out, _, _ = self.crop_square_fast(
            search_img, s_box,
            out_size=self.template_sz,
            scale_factor=self.template_scale_f,
            jitter_f=self.template_jitter_f)

        if np.random.rand() < 0.5 and not ('left' in s_lang and 'right' in s_lang):
            search_img, s_box = self.horizontal_flip(search_img, s_box)
            template_out = self.horizontal_flip(template_out)
            if 'left' in s_lang:
                s_lang = s_lang.replace('left', 'right')
            elif 'right' in s_lang:
                s_lang = s_lang.replace('right', 'left')

        if self.debug:
            print(t_box.astype(int), s_box.astype(int), s_lang)
            self.debug_fn([template_in, template_out, search_img], [t_box, s_box])

        template_in, template_out, search_img = map(lambda x: x.transpose(2, 0, 1).astype(np.float32),
                                                    [template_in, template_out, search_img])

        return template_in, search_img, template_out

    def debug_fn(self, im, box):  # [x, y, x, y]
        t_in = im[0]
        t_out = im[1]
        s_img = im[2]

        t_bbox = box[0]
        s_bbox = box[1]

        t_in = cv2.rectangle(
            t_in,
            (int(t_bbox[0]), int(t_bbox[1])),
            (int(t_bbox[0] + t_bbox[2] - 1), int(t_bbox[1] + t_bbox[3] - 1)), (0, 255, 0), 4)

        t_out = cv2.rectangle(
            t_out,
            (int(t_bbox[0]), int(t_bbox[1])),
            (int(t_bbox[0] + t_bbox[2] - 1), int(t_bbox[1] + t_bbox[3] - 1)), (0, 255, 0), 4)

        s_img = cv2.rectangle(
            s_img,
            (int(s_bbox[0]), int(s_bbox[1])),
            (int(s_bbox[0] + s_bbox[2] - 1), int(s_bbox[1] + s_bbox[3] - 1)), (0, 255, 0), 4)

        self.ax1.imshow(t_in)
        self.ax2.imshow(t_out)
        self.ax3.imshow(s_img)
        self.fig.show()
        plt.waitforbuttonpress()


def lmdb_translation_template_build_fn(cfg, lmdb, json):
    train_dataset = LMDBPatch(cfg, lmdb_path=lmdb, json_path=json,
                              dataset_name_list=cfg.datasets_train, num_samples=cfg.num_samples_train)
    if len(cfg.datasets_val) > 0:
        val_dataset = LMDBPatch(cfg, lmdb_path=lmdb, json_path=json,
                                dataset_name_list=cfg.datasets_val, num_samples=cfg.num_samples_val)
    else:
        val_dataset = None

    return train_dataset, val_dataset


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    from config import cfg_translation as cfg
    from register import path_register

    cfg.data.datasets_train = [
            'lasot_train',  # 1,120
            'vid_sent_train',  # 6,582
            'ytb2021ref_vos_train',  # 64,59
            'tnl2k_train',  # 1,300
        ]
    cfg.data.datasets_val = []

    trainset, valset = lmdb_translation_template_build_fn(cfg.data, lmdb=path_register.lmdb, json=path_register.json)

    train_loader = DataLoader(
        trainset,
        batch_size=1,
        num_workers=0,
        shuffle=True,
        sampler=None,
        drop_last=True,
        collate_fn=lmdb_translation_template_collate_fn
    )

    for i, image in enumerate(train_loader):
        print(i)
