import argparse
import os
import sys
import json

import numpy as np
from PIL import Image

sys.path.append('../')
from lib.dataset import LMDBData, SubSet, DatasetLoader, parse_loop

"""
Examples:

YouTube-VIS2021/
├── test
│     ├── instances.json
│     └── JPEGImages
├── test_submission_sample
│     └── results.json
├── train
│     ├── instances.json
│     └── JPEGImages
├── valid
│     ├── instances.json
│     └── JPEGImages
└── valid_submission_sample
    └── results.json


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


class YouTubeLoader(DatasetLoader):
    def __init__(self, name: str, split: str):
        super(YouTubeLoader, self).__init__()

        self.dataset_name = name
        self.object_id = None
        self.category_id = None

        if split == 'train' or split == 'train_all' :
            self.root = '/data2/Datasets/YouTube-VOS/YouTube-VIS2021/train'

        # Both val set and test set do not provide annotations, evaluated on codalab.
        # We split the train set into train sub-set and val sub-set.
        # See `ytb2021vis_train.txt` and `ytb2021vis_val.txt`. 5655/628 samples
        elif split == 'val':
            self.root = '/data2/Datasets/YouTube-VOS/YouTube-VIS2021/train'

        else:
            raise NotImplementedError

        self.data_dir = os.path.join(self.root, 'JPEGImages')

        if split == 'train_all' or split == 'train' or split == 'val':
            with open(os.path.join(self.root, 'instances.json'), 'r') as fin:
                self.meta = json.load(fin)

            if split == 'train_all':
                self.video_list = self.meta['annotations']
            elif split == 'train':
                idx = np.loadtxt('ytb2021vis_train.txt', delimiter='\n').astype(int)
                self.video_list = np.array(self.meta['annotations'])[idx]
            elif split == 'val':
                idx = np.loadtxt('ytb2021vis_val.txt', delimiter='\n').astype(int)
                self.video_list = np.array(self.meta['annotations'])[idx]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def get_video_info(self, anno_dict: dict):
        video_id = anno_dict['video_id']
        self.object_id = anno_dict['id']
        self.category_id = anno_dict['category_id']

        this_video = self.meta['videos'][video_id-1]
        assert this_video['id'] == video_id

        self.video_name = this_video['file_names'][0].split('/')[0]

        self.gt_list = np.array([[0, 0, 0, 0] if box is None else box for box in anno_dict['bboxes']])

        self.img_list = np.array(this_video['file_names'])
        self.key_list = np.array([fname.split('.')[0] for fname in self.img_list])

        self.lang_list = None

        self.mask_list = np.array([rle for rle in anno_dict['segmentations']])

        absent = np.array([1 if box is None else 0 for box in anno_dict['bboxes']])
        self.gt_list = self.gt_list[absent == 0]
        self.img_list = self.img_list[absent == 0]
        self.mask_list = self.mask_list[absent == 0]

        self.imw, self.imh = this_video['width'], this_video['height']


if __name__ == '__main__':
    # #################################
    # [ytb2021vis_train_all] -- 6283 videos, 175384 frames, done !
    # [ytb2021vis_train] -- 5655 videos, 157838 frames, done !
    # [ytb2021vis_val] -- 628 videos, 17546 frames, done !
    # #################################
    parser = argparse.ArgumentParser(description='parse youtube vis 2021 for lmdb')
    parser.add_argument('--split', default='val', type=str, choices=['train_all', 'train', 'val'],
                        help='select training set or testing set')
    parser.add_argument('--dir', default='/data3/LMDB', type=str)
    parser.add_argument('--only_json', dest='only_json', action='store_true', default=False)
    args = parser.parse_args()

    dataset_name = f'ytb2021vis_{args.split}'
    save_dir = os.path.join(args.dir, dataset_name)

    if args.only_json:
        lmdb_dict = None
        if os.path.exists(os.path.join(save_dir, f'{dataset_name}.json')):
            print(f'Directory not empty: {dataset_name}.json had been built.')
            sys.exit()
        else:
            os.makedirs(save_dir, exist_ok=True)
    else:
        lmdb_dict = LMDBData(save_dir=save_dir)
    data_info = SubSet(name=dataset_name)
    data_set = YouTubeLoader(name=dataset_name, split=args.split)

    parse_loop(name=dataset_name, save_dir=save_dir,
               data_set=data_set, json_obj=data_info, lmdb_obj=lmdb_dict)
