import argparse
import os
import sys
import json
import cv2
import numpy as np
from PIL import Image
import pycocotools.mask as mask_utils
sys.path.append('../')
from lib.dataset import LMDBData, SubSet, DatasetLoader, parse_loop
from tqdm import tqdm

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

        if split == 'train':
            self.root = '/data2/Datasets/YouTube-VOS/Refer-YouTube-VOS2021/train'
            self.anno_dir = os.path.join(self.root, 'Annotations')
            self.meta_exp_root = '/data2/Datasets/YouTube-VOS/Refer-YouTube-VOS2021/meta_expressions/train'

            # self.root = '/data2/Datasets/YouTube-VOS/YouTube-VOS2018/train'
            # self.anno_dir = os.path.join(self.root, 'Annotations')

        # Both val set and test set do not provide annotations, evaluated on codalab.
        # We split the train set into train sub-set and val sub-set.
        # See `ytb2021vis_train.txt` and `ytb2021vis_val.txt`. 5655/628 samples
        elif split == 'val':
            self.root = '/data2/Datasets/YouTube-VOS/Refer-YouTube-VOS2021/train'

        else:
            raise NotImplementedError

        self.data_dir = os.path.join(self.root, 'JPEGImages')

        if split == 'train':
            with open(os.path.join(self.root, 'meta.json'), 'r') as fin:
                meta = json.load(fin)['videos']
            with open(os.path.join(self.meta_exp_root, 'meta_expressions.json'), 'r') as fin:
                meta_exp = json.load(fin)['videos']

            vid_names = list(meta_exp.keys())

            self.exp_list = dict()
            for n in vid_names:
                for k, v in meta_exp[n]['expressions'].items():
                    tmp_name = n + f"_{v['obj_id']}"  # video_{exp_id}_{obj_id}
                    if tmp_name not in self.exp_list:
                        self.exp_list.update({tmp_name: [v['exp']]})
                    else:
                        self.exp_list[tmp_name].append(v['exp'])

            self.anno_dicts = meta
        else:
            raise NotImplementedError

        pre_object_video_list_dict = self.extract_tracks(vid_names)
        self.video_list = []
        for (k, v) in pre_object_video_list_dict.items():
            self.video_list.append(v)

    def extract_tracks(self, video_list):
        pre_object_video_list_dict = dict()

        for video_name in tqdm(video_list, "process each object"):

            mask_list = os.listdir(os.path.join(self.anno_dir, video_name))
            mask_list = sorted(mask_list)
            fname_list = [img_file.split('.')[0] for img_file in mask_list]
            mask_list = np.array([os.path.join(self.anno_dir, video_name, img_file) for img_file in mask_list])

            for mask, file_name in zip(mask_list, fname_list):
                # mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)  #
                # pixel values are not the same as the object_id, don't use cv2

                mask = Image.open(mask)
                mask = np.atleast_3d(mask)[..., 0]

                track_ids = np.unique(mask)
                for tid in track_ids:
                    if tid != 0:
                        tmp_mask = (mask == tid)
                        compressed_rle = mask_utils.encode(np.asfortranarray(tmp_mask))

                        if cv2.__version__[-5] == '4':
                            contours, _ = cv2.findContours(tmp_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                                           cv2.CHAIN_APPROX_NONE)
                        else:
                            _, contours, _ = cv2.findContours(tmp_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                                              cv2.CHAIN_APPROX_NONE)

                        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
                        contour = contours[np.argmax(cnt_area)]  # select the contour who has the max area
                        box = np.array(cv2.boundingRect(contour)).reshape(-1)  # axis_align_rectangle (x, y, w, h)

                        if f'{video_name}_{tid}' not in pre_object_video_list_dict:
                            pre_object_video_list_dict[f'{video_name}_{tid}'] = {
                                'video_name': video_name,
                                'bboxes': [box.tolist()],
                                'file_names': [f'{video_name}/{file_name}.jpg'],
                                'segmentations': [compressed_rle],
                                'width': mask.shape[1],
                                'height': mask.shape[0],
                                'language': self.exp_list[f'{video_name}_{tid}']
                            }
                        else:
                            pre_object_video_list_dict[f'{video_name}_{tid}']['bboxes'].append(box.tolist())
                            pre_object_video_list_dict[f'{video_name}_{tid}']['segmentations'].append(compressed_rle)
                            pre_object_video_list_dict[f'{video_name}_{tid}']['file_names'].append(
                                f'{video_name}/{file_name}.jpg')

        return pre_object_video_list_dict

    def get_video_info(self, anno_dict: dict):

        self.video_name = anno_dict['file_names'][0].split('/')[0]

        self.gt_list = np.array([[0, 0, 0, 0] if box is None else box for box in anno_dict['bboxes']])

        self.img_list = np.array(anno_dict['file_names'])
        self.key_list = np.array([fname.split('.')[0] for fname in self.img_list])

        self.lang_list = np.array([anno_dict['language'] for _ in range(self.gt_list.shape[0])])

        self.mask_list = np.array([rle for rle in anno_dict['segmentations']])

        absent = np.array([1 if box is None else 0 for box in anno_dict['bboxes']])
        self.gt_list = self.gt_list[absent == 0]
        self.img_list = self.img_list[absent == 0]
        self.mask_list = self.mask_list[absent == 0]

        self.imw, self.imh = anno_dict['width'], anno_dict['height']


if __name__ == '__main__':
    # #################################
    # [ytb2021ref_vos_train] -- 6459 videos, 159976 frames, done !
    # #################################
    parser = argparse.ArgumentParser(description='parse youtube referring vos 2021 for lmdb')
    parser.add_argument('--split', default='train', type=str, choices=['train'],
                        help='select training set or testing set')
    parser.add_argument('--dir', default='/home/space', type=str)
    parser.add_argument('--only_json', dest='only_json', action='store_true', default=False)
    args = parser.parse_args()

    dataset_name = f'ytb2021ref_vos_{args.split}'
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
