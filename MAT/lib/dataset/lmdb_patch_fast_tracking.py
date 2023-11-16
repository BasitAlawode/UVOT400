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


def lmdb_patchFT_collate_fn(batch):
    template_img = [torch.Tensor(item[0]).unsqueeze(0) for item in batch]
    search_img = [torch.Tensor(item[1]).unsqueeze(0) for item in batch]
    s_box = [torch.Tensor(item[2]).unsqueeze(0) for item in batch]
    s_lang = [item[3] for item in batch]

    template_img = torch.cat(template_img, dim=0)
    search_img = torch.cat(search_img, dim=0)
    s_box = torch.cat(s_box, dim=0)

    return {
        'template': template_img,
        'search': search_img,
        'target': s_box,
        'language': s_lang,
    }


class LMDBPatchFastTracking(BaseDataset):

    def __init__(self, cfg, lmdb_path, json_path, dataset_name_list: list = None, num_samples: int = None):
        super(LMDBPatchFastTracking).__init__()

        self.debug = False
        if self.debug:
            self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)

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
            # aug.ToGray(p=0.05),
            aug.GaussianBlur(blur_limit=0, sigma_limit=(0.2, 1), p=0.5),
            # aug.HorizontalFlip(p=0.5),
        ])

        # load dataset
        self.LMDB_ENVS = {}
        self.LMDB_HANDLES = {}
        self.dataset_list: List = []
        for name in dataset_name_list:
            dataset = SubSet(name=name, load=self.json_path[name])
            self.dataset_list.append([dataset.data_set, len(dataset.data_set)])

            env = lmdb.open(self.lmdb_path[name], readonly=True, lock=False, readahead=False, meminit=False)
            self.LMDB_ENVS[name] = env
            item = env.begin(write=False)
            self.LMDB_HANDLES[name] = item

        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        video_list, list_len = random.choice(self.dataset_list)
        idx = np.random.randint(0, list_len)

        t_dict, s_dict = self.check_sample(video_list[idx], video_list, self.sample_range)

        # read RGB image, [x y w h]
        template_img, t_box, t_lang = self.parse_frame_lmdb(t_dict, self.LMDB_HANDLES, need_language=True)
        _search_img, _s_box, s_lang = self.parse_frame_lmdb(s_dict, self.LMDB_HANDLES, need_language=True)

        template_img, _, t_box = self.crop_patch_fast(
            template_img, t_box,
            out_size=self.template_sz,
            scale_factor=self.template_scale_f,
            jitter_f=self.template_jitter_f)

        search_img, _, s_box = self.crop_patch_fast(
            _search_img, _s_box,
            out_size=self.search_sz,
            scale_factor=self.search_scale_f,
            jitter_f=self.search_jitter_f)

        template_img, search_img = map(lambda im: self.aug(image=im)["image"], [template_img, search_img])

        if np.random.rand() < 0.5:
            search_img, s_box = self.horizontal_flip(search_img, s_box)
            if 'left' in s_lang:
                s_lang = s_lang.replace('left', 'right')
            elif 'right' in s_lang:
                s_lang = s_lang.replace('right', 'left')

        if self.debug:
            print(t_box.astype(int), s_box.astype(int), s_lang)
            self.debug_fn([template_img, search_img], [t_box, s_box])

        template_img, search_img = map(lambda x: x.transpose(2, 0, 1).astype(np.float32), [template_img, search_img])
        t_box, s_box = map(lambda x: x.astype(np.float32), [t_box, s_box])  # [x, y, w, h]

        # [x, y, w, h] -> norm[x, y, x, y]
        s_box[2:] = s_box[:2] + s_box[2:] - 1
        s_box[0::2] = s_box[0::2] / self.search_sz[1]
        s_box[1::2] = s_box[1::2] / self.search_sz[0]

        return template_img, search_img, s_box, s_lang

    def debug_fn(self, im, box):  # [x, y, x, y]
        t_img = im[0]
        s_img = im[1]

        t_bbox = box[0]
        s_bbox = box[1]

        t_img = cv2.rectangle(
            t_img,
            (int(t_bbox[0]), int(t_bbox[1])),
            (int(t_bbox[0] + t_bbox[2] - 1), int(t_bbox[1] + t_bbox[3] - 1)), (0, 255, 0), 4)

        s_img = cv2.rectangle(
            s_img,
            (int(s_bbox[0]), int(s_bbox[1])),
            (int(s_bbox[0] + s_bbox[2] - 1), int(s_bbox[1] + s_bbox[3] - 1)), (0, 255, 0), 4)

        self.ax1.imshow(t_img)
        self.ax2.imshow(s_img)
        self.fig.show()
        plt.waitforbuttonpress()


def lmdb_patchFT_build_fn(cfg, lmdb, json):
    train_dataset = LMDBPatchFastTracking(cfg, lmdb_path=lmdb, json_path=json,
                                  dataset_name_list=cfg.datasets_train, num_samples=cfg.num_samples_train)
    if len(cfg.datasets_val) > 0:
        val_dataset = LMDBPatchFastTracking(cfg, lmdb_path=lmdb, json_path=json,
                                    dataset_name_list=cfg.datasets_val, num_samples=cfg.num_samples_val)
    else:
        val_dataset = None

    return train_dataset, val_dataset


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    from config import cfg_translation_track as cfg
    from register import path_register

    cfg.data.datasets_train = ['got10k_train', 'got10k_val']
    cfg.data.datasets_val = []

    trainset, valset = lmdb_patchFT_build_fn(cfg.data, lmdb=path_register.lmdb, json=path_register.json)

    train_loader = DataLoader(
        trainset,
        batch_size=1,
        num_workers=0,
        shuffle=True,
        sampler=None,
        drop_last=True,
        collate_fn=lmdb_patchFT_collate_fn
    )

    for i, image in enumerate(train_loader):
        print(i)
