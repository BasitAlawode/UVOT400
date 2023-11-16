import argparse
import os
import sys

import numpy as np

sys.path.append('../')
from lib.dataset import LMDBData, SubSet, DatasetLoader, parse_loop

"""
Examples:

GOT-10k/
├── test
├── train
├── val
└── vot19-20_got10k_prohibited_1000.txt


frame_dict_list = [
    {'bbox': [409.0, 428.0, 1509.0, 312.0], 
    'dataset': 'got10k_train-vot', 
    'key': 'GOT-10k_Train_001281/00000007', 
    'length': 100, 
    'path': 'GOT-10k_Train_001281/00000007.jpg', 
    'size': [1920, 1080], 
    'video': 'GOT-10k_Train_001281'}, 
     ...
]
"""


class GOT10kLoader(DatasetLoader):
    def __init__(self, name: str, split: str):
        super(GOT10kLoader, self).__init__()

        self.dataset_name = name

        if 'train' in split:
            self.root = '/data1/Datasets/GOT-10k/train_depth'
        else:
            raise NotImplementedError

        self.data_dir = self.root

        if split == 'train':
            video_list = os.listdir(self.root)
            self.video_list = sorted(video_list)
        else:
            raise NotImplementedError

    def get_video_info(self, video_name: str):
        self.video_name = video_name
        video_dir = os.path.join(self.data_dir, video_name)

        img_list = sorted(os.listdir(video_dir))
        img_list = np.array([os.path.join(video_name, img_file) for img_file in img_list if '.png' in img_file])
        self.img_list = img_list
        self.key_list = np.array([f.split('.')[0] for f in self.img_list])

        self.lang_list = None

        self.imw, self.imh = None, None


if __name__ == '__main__':
    # #################################
    # [got10k_train] -- 9335 videos, 1401877 frames, done !
    # [got10k_train_vot] -- 8335 videos, 1250652 frames, done !
    # [got10k_val] -- 180 videos, 20979 frames, done !
    # #################################
    parser = argparse.ArgumentParser(description='parse got10k for lmdb')
    parser.add_argument('split', default='train', type=str, choices=['train', 'train_vot', 'val'],
                        help='select training set or testing set')
    parser.add_argument('--dir', default='/data3/LMDB', type=str)
    parser.add_argument('--only_json', dest='only_json', action='store_true', default=False)
    args = parser.parse_args()

    dataset_name = f'got10k_depth_{args.split}'
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
    data_set = GOT10kLoader(name=dataset_name, split=args.split)

    parse_loop(name=dataset_name, save_dir=save_dir,
               data_set=data_set, json_obj=data_info, lmdb_obj=lmdb_dict)
