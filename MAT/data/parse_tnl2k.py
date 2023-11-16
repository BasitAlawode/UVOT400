import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.append('../')
from lib.dataset import LMDBData, SubSet, DatasetLoader, parse_loop

"""
Examples:

TNL2K/
├── TNL2K_test_subset
└── TNL2K_train_subset


frame_dict_list = [
    {'bbox': [285.0, 168.0, 19.0, 32.0], 
    'dataset': 'tnl2k_train', 
    'key': 'CM_manfaraway_done/00054i', 
    'language': 'the man walking on the lawn ', 
    'length': 249, 
    'path': 'CM_manfaraway_done/imgs/00054i.jpg', 
    'size': [630, 460], 
    'video': 'CM_manfaraway_done'}, 
     ...
]
"""


class TNL2KLoader(DatasetLoader):
    def __init__(self, name: str, split: str):
        super(TNL2KLoader, self).__init__()

        self.dataset_name = name

        if 'train' in split:
            self.root = '/data2/Datasets/TNL2K/TNL2K_train_subset'
        else:
            self.root = '/data2/Datasets/TNL2K/TNL2K_test_subset'

        self.data_dir = self.root

        if split == 'train' or split == 'test':
            video_list = os.listdir(self.root)
            self.video_list = sorted(video_list)
        else:
            raise NotImplementedError

    def get_video_info(self, video_name: str):
        self.video_name = video_name
        video_dir = os.path.join(self.data_dir, video_name)

        gt_list = np.loadtxt(os.path.join(video_dir, 'groundtruth.txt'), delimiter=',')
        self.gt_list = gt_list.reshape(-1, 4)

        img_list = sorted(os.listdir(os.path.join(video_dir, 'imgs')))
        img_list = np.array([os.path.join(video_name, 'imgs', img_file) for img_file in img_list])
        self.img_list = img_list[:gt_list.shape[0]]  # used as the key of lmdb
        self.key_list = np.array([os.path.join(self.video_name, f.split('/')[-1].split('.')[0])
                                  for f in self.img_list])

        lang_path = os.path.join(video_dir, 'language.txt')
        with open(lang_path, 'r') as sf:
            lang = sf.readline()
        self.lang_list = np.array([lang for _ in range(self.gt_list.shape[0])])

        present = np.logical_and(gt_list[:, 2] > 0, gt_list[:, 3] > 0)
        self.gt_list = self.gt_list[present]
        self.img_list = self.img_list[present]
        self.lang_list = self.lang_list[present]

        img = Image.open(os.path.join(self.data_dir, self.img_list[0]))
        self.imw, self.imh = img.size


if __name__ == '__main__':
    # #################################
    # [tnl2k_train] -- 1300 videos, 650913 frames, done !
    # [tnl2k_test] -- 700 videos, 450773 frames, done !
    # #################################
    parser = argparse.ArgumentParser(description='parse tnl2k for lmdb')
    parser.add_argument('split', default='train', type=str, choices=['train', 'test'],
                        help='select training set or testing set')
    parser.add_argument('--dir', default='/data3/LMDB', type=str)
    parser.add_argument('--only_json', dest='only_json', action='store_true', default=False)
    args = parser.parse_args()

    dataset_name = f'tnl2k_{args.split}'
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
    data_set = TNL2KLoader(name=dataset_name, split=args.split)

    parse_loop(name=dataset_name, save_dir=save_dir,
               data_set=data_set, json_obj=data_info, lmdb_obj=lmdb_dict)
