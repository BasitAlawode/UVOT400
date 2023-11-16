import cv2
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import lmdb
import albumentations as aug

from lib.dataset._dataset import SubSet, BaseDataset

"""
Examples:

frame_dict_list = [
    {'bbox': [29.0, 56.0, 935.0, 555.0], 
    'dataset': 'ytb2021vis_train', 
    'key': '8dcccd2bd2/00040', 
    'length': 36, 
    'mask': {'counts': [21179, 29, 691, ..., 15, 228107], 'size': [720, 1280]},  # uncompressed RLE
    'path': '8dcccd2bd2/00040.jpg', 
    'size': [1280, 720], 
    'video': '8dcccd2bd2'}, 
     ...
]
"""


class LMDBPatch(BaseDataset):

    def __init__(self, cfg, lmdb_path, json_path, dataset_name_list: list = None, num_samples: int = None):
        super(LMDBPatch).__init__()

        self.fig, self.ax = plt.subplots()

        self.lmdb_path: dict = lmdb_path
        self.json_path: dict = json_path

        self.sample_range: int = cfg.sample_range

        self.search_sz: int = cfg.search_size
        self.search_scale_f: float = cfg.search_scale_f
        self.search_jitter_f: List[float, float] = cfg.search_jitter_f

        self.template_sz: int = cfg.template_size
        self.template_scale_f: float = cfg.template_scale_f
        self.template_jitter_f: List[float, float] = cfg.template_jitter_f

        # Declare an augmentation pipeline
        self.aug = aug.Compose([
            aug.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1),
            aug.ToGray(p=0.05),
            # aug.GaussianBlur(blur_limit=0, sigma_limit=(0.2, 1), p=0.5),
            # aug.HorizontalFlip(p=0.5),
        ])

        # load dataset
        self.LMDB_ENVS = {}
        self.LMDB_HANDLES = {}
        self.video_list: List = []
        for name in dataset_name_list:
            env = lmdb.open(self.lmdb_path[name], readonly=True, lock=False, readahead=False, meminit=False)
            self.LMDB_ENVS[name] = env
            item = env.begin(write=False)
            self.LMDB_HANDLES[name] = item

            dataset = SubSet(name=name, load=self.json_path[name])
            if 'coco' in name:
                random.shuffle(dataset.data_set)
                self.video_list += dataset.data_set[:10000]
            elif 'refcocos_train' in name:  # 300k+
                random.shuffle(dataset.data_set)
                self.video_list += dataset.data_set[:]
            elif 'vg100k_train' in name:  # 5000k+
                random.shuffle(dataset.data_set)
                self.video_list += dataset.data_set[:]
            else:
                self.video_list += dataset.data_set

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
        t_dict, s_dict = self.check_sample(self.video_list[item][::15], self.video_list, self.sample_range)

        # read RGB image, [x y w h]
        img, box, lang, mask = self.parse_frame_lmdb_mask(s_dict, self.LMDB_HANDLES)

        if mask is not None:
            assert mask.shape == img.shape[:2]
            img = img.astype(float)
            img[:, :, 2] = img[:, :, 2] * (1-mask) + 255 * mask #+ img[:, :, 2] * mask * 0.2
            img = img.astype(np.uint8)

        print(lang)
        t_img = cv2.rectangle(
            img,
            (int(box[0]), int(box[1])),
            (int(box[0] + box[2] - 1), int(box[1] + box[3] - 1)), (0, 255, 0), 4)

        self.ax.imshow(t_img)
        self.fig.show()
        plt.waitforbuttonpress()


if __name__ == '__main__':
    import random

    from config.cfg_baseline import cfg as settings

    settings.data.datasets_train = [
        # 'tnl2k_train',
        # 'tnl2k_test',
        # 'got10k_train',
        # 'got10k_train-vot',
        # 'got10k_val',
        # 'lasot_train',
        # 'ytb2021vis_train',
        'trackingnet_train_p0',
        # 'trackingnet_train_p1',
        # 'trackingnet_train_p2',
        # 'vid_sent_train',
    ]

    settings.data.datasets_val = []

    lmdb_path = {'tnl2k_train': '/data3/LMDB/tnl2k_train/',
                 'tnl2k_test': '/data3/LMDB/tnl2k_test/',
                 'got10k_train': '/data3/LMDB/got10k_train/',
                 'got10k_train-vot': '/data3/LMDB/got10k_train/',
                 'got10k_val': '/data3/LMDB/got10k_val/',
                 'lasot_train': '/data3/LMDB/lasot_train/',
                 'ytb2021vis_train': '/data3/LMDB/ytb2021vis_train/',
                 'trackingnet_train_p0': '/data3/LMDB/trackingnet_train_p0/',
                 'trackingnet_train_p1': '/data3/LMDB/trackingnet_train_p1/',
                 'trackingnet_train_p2': '/data3/LMDB/trackingnet_train_p2/',
                 'vid_sent_train': '/data3/LMDB/vid_sent_train/',
                 }

    json_path = {'tnl2k_train': '/data3/LMDB/tnl2k_train/tnl2k_train.json',
                 'tnl2k_test': '/data3/LMDB/tnl2k_test/tnl2k_test.json',
                 'got10k_train': '/data3/LMDB/got10k_train/got10k_train.json',
                 'got10k_train-vot': '/data3/LMDB/got10k_train-vot/got10k_train-vot.json',
                 'got10k_val': '/data3/LMDB/got10k_val/got10k_val.json',
                 'lasot_train': '/data3/LMDB/lasot_train/lasot_train.json',
                 'ytb2021vis_train': '/data3/LMDB/ytb2021vis_train/ytb2021vis_train.json',
                 'trackingnet_train_p0': '/data3/LMDB/trackingnet_train_p0/trackingnet_train_p0.json',
                 'trackingnet_train_p1': '/data3/LMDB/trackingnet_train_p1/trackingnet_train_p1.json',
                 'trackingnet_train_p2': '/data3/LMDB/trackingnet_train_p2/trackingnet_train_p2.json',
                 'vid_sent_train': '/data3/LMDB/vid_sent_train/vid_sent_train.json',
                 }

    train_set = LMDBPatch(settings.data, lmdb_path=lmdb_path, json_path=json_path,
                          dataset_name_list=settings.data.datasets_train,
                          num_samples=settings.data.num_samples_train)

    length = list(range(train_set.__len__()))
    random.shuffle(length)

    for idx in length:
        train_set.__getitem__(idx)
