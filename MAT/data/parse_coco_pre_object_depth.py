import argparse
import os
import sys
import json
from collections import defaultdict

import numpy as np
from PIL import Image

sys.path.append('../')
from lib.dataset import LMDBData, SubSet, DatasetLoader, parse_loop

"""
Examples:

./COCO
├── annotations
│     ├──instances_train2017.json
│     └── instances_val2017.json
├── train2014
├── train2017
├── val2017
└── val2017.txt



frame_dict_list = [
    {'path': 'train2017/000000432732.jpg', 
    'bbox': [91.01, 89.24, 388.85, 268.92], 
    'key': 'train2017/000000432732', 
    'dataset': 'coco_train', 
    'video': 'train2017', 
    'length': 1, 
    'size': [640, 428], 
    'mask': array([list([327.26, 354.04, ..., 354.04]),
       list([372.23, 342.42, ..., 342.42]),
       list([349.0, 308.69, ..., 308.69]),
       list([366.11, 230.42, ..., 230.42])], dtype=object)}  # polygon
]

"""


class COCOLoader(DatasetLoader):
    def __init__(self, name: str, split: str):
        super(COCOLoader, self).__init__()

        self.dataset_name = name
        self.object_id = None
        self.category_id = None

        if split == 'train':
            self.root = '/data1/Datasets/COCO/'
        else:
            raise NotImplementedError

        self.data_dir = self.root

        if split == 'train':

            with open(os.path.join(self.root, 'annotations', 'instances_train2017.json'), 'r') as fin:
                self.meta = json.load(fin)
            self.video_list = self.meta['annotations']

        elif split == 'val':

            with open(os.path.join(self.root, 'annotations', 'instances_val2017.json'), 'r') as fin:
                self.meta = json.load(fin)
            self.video_list = self.meta['annotations']
        else:
            raise NotImplementedError

        self.img_ann_map = defaultdict(list)
        self.cat_img_map = defaultdict(list)
        self.anns = {}
        self.cats = {}
        self.imgs = {}

        for ann in self.meta["annotations"]:
            self.img_ann_map[ann["image_id"]].append(ann)
            self.anns[ann["id"]] = ann

        for img in self.meta["images"]:
            self.imgs[img["id"]] = img

        for cat in self.meta["categories"]:
            self.cats[cat["id"]] = cat

        for ann in self.meta["annotations"]:
            self.cat_img_map[ann["category_id"]].append(ann["image_id"])

        self.video_list = list(self.anns.keys())

    def get_video_info(self, id: int):

        anno_dict = self.anns[id]
        image_id = anno_dict['image_id']

        # self.object_id = id
        # self.category_id = anno_dict['category_id']

        self.video_name = self.imgs[image_id]['coco_url'].split('/')[-2]
        # coco_url: (e.g., http://images.cocodataset.org/train2017/000000391895.jpg)

        # self.gt_list = np.array([anno_dict['bbox']])

        self.img_list = np.array([os.path.join(self.imgs[image_id]['coco_url'].split('/')[-2],
                                               self.imgs[image_id]['coco_url'].split('/')[-1])])
        self.key_list = np.array([f.split('.')[0] for f in self.img_list])

        self.lang_list = None

        # self.mask_list = [anno_dict['segmentation']]  # [polygon]

        self.imw, self.imh = self.imgs[image_id]['width'], self.imgs[image_id]['height']


if __name__ == '__main__':
    # #################################
    # [coco2017_train] -- 860001 videos, 860001 frames, done !
    # [coco2017_val] -- 36781 videos, 36781 frames, done !
    # #################################
    parser = argparse.ArgumentParser(description='parse COCO2017 for lmdb')
    parser.add_argument('--split', default='train', type=str, choices=['train', 'val'],
                        help='select training set or testing set')
    parser.add_argument('--dir', default='/data3/LMDB', type=str)
    parser.add_argument('--only_json', dest='only_json', action='store_true', default=False)
    args = parser.parse_args()

    dataset_name = f'coco2017_{args.split}'
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
    data_set = COCOLoader(name=dataset_name, split=args.split)

    parse_loop(name=dataset_name, save_dir=save_dir,
               data_set=data_set, json_obj=data_info, lmdb_obj=lmdb_dict)
