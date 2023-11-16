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


class LaSOTLoader(DatasetLoader):
    def __init__(self, name: str, split: str):
        super(LaSOTLoader, self).__init__()

        self.dataset_name = name

        self.root = '/data1/Datasets/LaSOT/'
        self.anno_dir = os.path.join(self.root, 'annos/absent')
        self.data_dir = os.path.join(self.root, 'depths')

        if split == 'train':
            with open('/data1/Datasets/LaSOT/training_set.txt', 'r') as f:
                video_list = f.readlines()
            self.video_list = sorted([x.split('\n')[0] for x in video_list])
        elif split == 'test':
            with open('/data1/Datasets/LaSOT/testing_set.txt', 'r') as f:
                video_list = f.readlines()
            self.video_list = sorted([x.split('\n')[0] for x in video_list])
        else:
            raise NotImplementedError

    def get_video_info(self, video_name: str):
        self.video_name = video_name
        video_dir = os.path.join(self.data_dir, video_name)

        img_list = sorted(os.listdir(os.path.join(video_dir, 'depth')))
        img_list = np.array([os.path.join(video_name, 'depth', img_file) for img_file in img_list])
        self.img_list = img_list
        self.key_list = np.array([os.path.join(self.video_name, f.split('/')[-1].split('.')[0])
                                  for f in self.img_list])

        img = Image.open(os.path.join(self.data_dir, self.img_list[0]))
        self.imw, self.imh = img.size


if __name__ == '__main__':
    # #################################
    # [lasot_train] -- 1120 videos, 2752359 frames, done !
    # [lasot_test] -- 280 videos, 667014 frames, done !
    # #################################
    parser = argparse.ArgumentParser(description='parse lasot for lmdb')
    parser.add_argument('split', default='train', type=str, choices=['train', 'test'],
                        help='select training set or testing set')
    parser.add_argument('--dir', default='/data3/LMDB', type=str)
    parser.add_argument('--only_json', dest='only_json', action='store_true', default=False)
    args = parser.parse_args()

    dataset_name = f'lasot_depth_{args.split}'
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
    data_set = LaSOTLoader(name=dataset_name, split=args.split)

    parse_loop(name=dataset_name, save_dir=save_dir,
               data_set=data_set, json_obj=data_info, lmdb_obj=lmdb_dict)
