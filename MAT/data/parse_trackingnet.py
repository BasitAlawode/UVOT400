import argparse
import os
import sys
from typing import List
import numpy as np
from PIL import Image

sys.path.append('../')
from lib.dataset import LMDBData, SubSet, DatasetLoader, parse_loop

"""
Examples:

TrackingNet/
├── TEST
├── TRAIN_0
├── TRAIN_1
├── TRAIN_10
├── TRAIN_11
├── TRAIN_2
├── TRAIN_3
├── TRAIN_4
├── TRAIN_5
├── TRAIN_6
├── TRAIN_7
├── TRAIN_8
└── TRAIN_9

frame_dict_list = [
    {'bbox': [117.82, 53.04, 324.73, 217.20000000000002], 
    'dataset': 'trackingnet_train_p2', 
    'key': 'e1ZNGYPt280_0/329', 
    'length': 450, 
    'path': 'e1ZNGYPt280_0/329.jpg', 
    'size': [640, 352], 
    'video': 'e1ZNGYPt280_0'}, 
     ...
]
"""


class TrackingNetLoader(DatasetLoader):
    def __init__(self, name: str, split: str, parts: List[int]):
        super(TrackingNetLoader, self).__init__()

        self.dataset_name = name

        self.root = '/data1/Datasets/TrackingNet'

        if split == 'train':
            self.data_dir_list = []
            self.anno_dir_list = []
            self.video_list_list = []
            self.video_list = []
            for p in parts:
                dir_set = f'TRAIN_{p}'
                data_dir = os.path.join(self.root, dir_set, 'frames')
                anno_dir = os.path.join(self.root, dir_set, 'anno')

                self.data_dir_list.append(data_dir)
                self.anno_dir_list.append(anno_dir)

                video_list = sorted(os.listdir(data_dir))
                self.video_list_list.append(video_list)
                self.video_list += video_list
        else:
            raise NotImplementedError
        print()

    def get_video_info(self, video_name: str):
        for which in range(len(self.video_list_list)):
            if video_name in self.video_list_list[which]:
                break

        self.data_dir = self.data_dir_list[which]
        self.anno_dir = self.anno_dir_list[which]

        self.video_name = video_name
        video_dir = os.path.join(self.data_dir, video_name)

        gt_list = np.loadtxt(os.path.join(self.anno_dir, '{}.txt'.format(video_name)), delimiter=',')
        self.gt_list = gt_list.reshape(-1, 4)
        self.gt_list = np.clip(self.gt_list, 0, 99999)

        img_list = os.listdir(video_dir)
        img_list.sort(key=lambda x: int(x[:-4]))

        img_list = np.array([os.path.join(video_name, img_file) for img_file in img_list])
        self.img_list = img_list[:gt_list.shape[0]]  # used as the key of lmdb
        self.key_list = np.array([f.split('.')[0] for f in self.img_list])

        self.lang_list = None

        img = Image.open(os.path.join(self.data_dir, self.img_list[0]))
        self.imw, self.imh = img.size


if __name__ == '__main__':
    # #################################
    # [trackingnet_train_p0] -- 10044 videos, 4719996 frames, done !
    # [trackingnet_train_p1] -- 10038 videos, 4734664 frames, done !
    # [trackingnet_train_p2] -- 9261 videos, 4374939 frames, done !
    # #################################
    parser = argparse.ArgumentParser(description='parse got10k for lmdb')
    parser.add_argument('split', default='train', type=str, choices=['train', ],
                        help='select training set or testing set')
    parser.add_argument('--dir', default='/data3/LMDB', type=str)
    parser.add_argument('--only_json', dest='only_json', action='store_true', default=False)
    args = parser.parse_args()

    parts = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]

    for i_p in [0, 1, 2]:

        dataset_name = f'trackingnet_train_p{i_p}'
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
        data_set = TrackingNetLoader(name=dataset_name, split=args.split, parts=parts[i_p])

        parse_loop(name=dataset_name, save_dir=save_dir,
                   data_set=data_set, json_obj=data_info, lmdb_obj=lmdb_dict)
