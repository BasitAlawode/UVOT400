import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.append('../')
from lib.dataset import LMDBData, SubSet, DatasetLoader, parse_loop

"""
Examples:

dataset/
├── annos
├── images
├── testing_set.txt
└── training_set.txt


frame_dict_list = [
    {'path': 'airplane-10/img/00000001.jpg', 
    'bbox': [610.0, 341.0, 136.0, 28.0], 
    'language': 'airplane flying in the air', 
    'key': 'airplane-10/00000000', 
    'dataset': 'lasot_train', 
    'video': 'airplane-10', 
    'length': 1568, 
    'size': [1280, 720]}, 
     ...
]
"""


class IN1kLoader(DatasetLoader):
    def __init__(self, name: str, split: str):
        super(IN1kLoader, self).__init__()

        self.dataset_name = name

        self.root = '/data1/Datasets/ILSVRC2012/'

        if split == 'train':
            self.video_list = []
            self.data_dir = os.path.join(self.root, 'train')
            dir1 = sorted(os.listdir(self.data_dir))
            for d1 in dir1:
                ims = sorted(os.listdir(os.path.join(self.data_dir, d1)))
                self.video_list += [os.path.join(d1, im) for im in ims]

        elif split == 'val':
            self.video_list = []
            self.data_dir = os.path.join(self.root, 'val')
            ims = sorted(os.listdir(self.data_dir))
            self.video_list += ims
        else:
            raise NotImplementedError

    def get_video_info(self, img_path: str):
        self.video_name = img_path.split('.')[0]

        self.gt_list = None

        self.img_list = [img_path]
        self.key_list = [self.video_name]

        self.lang_list = None

        img = Image.open(os.path.join(self.data_dir, self.img_list[0]))
        self.imw, self.imh = img.size


if __name__ == '__main__':
    # #################################
    # [imagenet1k_train] -- 1281167 videos, 1281167 frames, done !
    # #################################
    parser = argparse.ArgumentParser(description='parse Imagenet1k for lmdb')
    parser.add_argument('--split', default='val', type=str, choices=['train', 'val'],
                        help='select training set or testing set')
    parser.add_argument('--dir', default='/home/space/', type=str)
    parser.add_argument('--only_json', dest='only_json', action='store_true', default=False)
    args = parser.parse_args()

    dataset_name = f'imagenet1k_{args.split}'
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
    data_set = IN1kLoader(name=dataset_name, split=args.split)

    parse_loop(name=dataset_name, save_dir=save_dir,
               data_set=data_set, json_obj=data_info, lmdb_obj=lmdb_dict)
