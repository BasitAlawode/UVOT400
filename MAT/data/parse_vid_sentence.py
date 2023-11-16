import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.append('../')
from lib.dataset import LMDBData, SubSet, DatasetLoader, parse_loop
from data.vid_sentence.vidDatasetParser import vidInfoParser

"""
Examples:

vid_sentence/
├── data
│     ├── ILSVRC
│     │     ├── Annotations
│     │     └── Data
│     └── VID
│         ├── test -> /data1/Datasets/ILSVRC2015/Data/VID/val/
│         ├── train -> /data1/Datasets/ILSVRC2015/Data/VID/train/
│         └── val -> /data1/Datasets/ILSVRC2015/Data/VID/val/
├── datasetUtils.py
├── __init__.py
├── LICENSE
├── README.md
├── vidDatasetParser.py
└── vis_instance.sh


frame_dict_list = [
    {'path': 'ILSVRC2015_train_01073002/000000.JPEG', 
    'bbox': [586, 168, 504, 332], 
    'key': 'ILSVRC2015_train_01073002/000000', 
    'language': ' A squirrel hides half of its body in the tree', 
    'dataset': 'vid_sent_train', 
    'video': 'ILSVRC2015_train_01073002', 
    'length': 426, 
    'size': [1280, 720]}, 
     ...
]
"""


class VIDSentLoader(DatasetLoader):
    def __init__(self, name: str, split: str):
        super(VIDSentLoader, self).__init__()

        self.dataset_name = name

        if 'train' in split:
            self.vid_parser = vidInfoParser(split, annFd=os.path.join('./vid_sentence/data/ILSVRC'))
            self.vid_parser.jpg_folder = os.path.join('./vid_sentence/data/VID/', split)
        elif 'val' in split:
            self.vid_parser = vidInfoParser(split, annFd=os.path.join('./vid_sentence/data/ILSVRC'))
            self.vid_parser.jpg_folder = os.path.join('./vid_sentence/data/VID/', split)
        else:
            raise NotImplementedError

        self.data_dir = self.vid_parser.jpg_folder

        use_key_index = list(self.vid_parser.tube_cap_dict.keys())
        self.video_list = sorted(use_key_index)

    def get_video_info(self, index: int):
        ann, _ = self.vid_parser.get_shot_anno_from_index(index)
        img_list, video_name = self.vid_parser.get_shot_frame_list_from_index(index)

        self.video_name = video_name

        gt_list = np.array([ann['track'][im_i]['bbox'] for im_i in range(len(img_list))])
        self.gt_list = gt_list.reshape(-1, 4).astype(np.float)
        self.gt_list[:, 2:] = self.gt_list[:, 2:] - self.gt_list[:, :2] + 1

        img_list = sorted(img_list)
        img_list = np.array([os.path.join(video_name, img_file + '.JPEG') for img_file in img_list])
        self.img_list = img_list[:gt_list.shape[0]]  # used as the key of lmdb
        self.key_list = np.array([fname.split('.')[0] for fname in self.img_list])

        lang = self.vid_parser.tube_cap_dict[index][0]
        self.lang_list = np.array([lang for _ in range(self.gt_list.shape[0])])

        # absent = np.array([1 if box is None else 0 for box in anno_dict['bboxes']])
        # self.gt_list = self.gt_list[absent == 0]
        # self.img_list = self.img_list[absent == 0]

        img = Image.open(os.path.join(self.data_dir, self.img_list[0]))
        self.imw, self.imh = img.size


if __name__ == '__main__':
    # #################################
    # [vid_sent_train] -- 6582 videos, 1467516 frames, done !
    # [vid_sent_val] -- 536 videos, 109630 frames, done !
    # #################################
    parser = argparse.ArgumentParser(description='parse VID sentence for lmdb')
    parser.add_argument('split', default='train', type=str, choices=['train', 'val'],
                        help='select training set or testing set')
    parser.add_argument('--dir', default='/data3/LMDB', type=str)
    parser.add_argument('--only_json', dest='only_json', action='store_true', default=False)
    args = parser.parse_args()

    dataset_name = f'vid_sent_{args.split}'
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
    data_set = VIDSentLoader(name=dataset_name, split=args.split)

    parse_loop(name=dataset_name, save_dir=save_dir,
               data_set=data_set, json_obj=data_info, lmdb_obj=lmdb_dict)
