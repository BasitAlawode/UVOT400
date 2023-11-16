import os
import sys
import cv2
import json

if 'jpeg4py' in sys.modules:
    import jpeg4py
import numpy as np
from typing import List

import torch
from torch.utils.data import Dataset
import pycocotools.mask as mask_utils


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return obj.decode()
        else:
            return super(NpEncoder, self).default(obj)


class SubSet(object):
    def __init__(self, name, load=None):
        self.data_set = []
        self.name = name
        self.num = 0
        self.length = 0  # number of frame / total length

        self.set_index = None
        self.set_num = None

        if load is not None:
            self.load(path=load)
            self.set_index = np.arange(self.num)
            self.set_num = self.num

    def load(self, path):
        with open(path) as fin:
            tmp = json.load(fin)
        self.data_set = tmp['data_set']  # list
        self.name = tmp['name']
        self.num = tmp['num']
        self.length = tmp['length']

    def create_set(self, num):
        self.set_num = num
        self.set_index = np.arange(num) % self.num
        np.random.shuffle(self.set_index)
        print('==> create datasets {} {}/{}'.format(self.name, self.set_num, self.num))

    def append(self, video_dict_list):
        self.data_set.append(video_dict_list)
        self.length += len(video_dict_list)
        self.num += 1

    def save(self, path):
        tmp = {
            'data_set': self.data_set,
            'name': self.name,
            'num': self.num,
            'length': self.length
        }
        json.dump(
            tmp,
            open(os.path.join(path, '{}.json'.format(self.name)), 'w'),
            indent=4,
            sort_keys=True,
            cls=NpEncoder,
        )
        print('{}.json has been saved in {}'.format(self.name, path))
        print('{} videos, {} frames'.format(self.num, self.length))


class BaseDataset(Dataset):

    def __init__(self):
        super().__init__()

        self.pytorch_mean = np.array([0.485, 0.456, 0.406])  # RGB
        self.pytorch_std = np.array([0.229, 0.224, 0.225])  # RGB

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def update_hyper_params(self, hp):
        if hp is not None:
            for key, value in hp.items():
                setattr(self, key, value)

    def check_sample(self, frame_list, video_list, sample_range, min_sz=1):
        anchor_frame_id = np.random.randint(0, len(frame_list), 1)[0]

        lp = max(anchor_frame_id - sample_range, 0)
        rp = min(anchor_frame_id + sample_range, len(frame_list))

        search_frame_id = np.random.randint(lp, rp, 1)[0]

        f_dict1: dict = frame_list[anchor_frame_id]
        x, y, w1, h1 = f_dict1['bbox']
        f_dict2: dict = frame_list[search_frame_id]
        x, y, w2, h2 = f_dict2['bbox']

        while np.min([w1, h1, w2, h2]) < min_sz:
            frame_list = video_list[np.random.randint(0, len(video_list))]

            anchor_frame_id = np.random.randint(0, len(frame_list), 1)[0]

            lp = max(anchor_frame_id - sample_range, 0)
            rp = min(anchor_frame_id + sample_range, len(frame_list))

            search_frame_id = np.random.randint(lp, rp, 1)[0]

            f_dict1: dict = frame_list[anchor_frame_id]
            x, y, w1, h1 = f_dict1['bbox']
            f_dict2: dict = frame_list[search_frame_id]
            x, y, w2, h2 = f_dict2['bbox']

        return f_dict1, f_dict2

    @staticmethod
    def parse_frame_dict(f_dict, data_path):
        f_path = f_dict['path']
        x, y, w, h = f_dict['bbox']
        dataset_name = f_dict['name']

        _path = os.path.join(data_path[dataset_name], f_path)

        if 'jpeg4py' in sys.modules:
            img = jpeg4py.JPEG(_path).decode()  # RGB
        else:
            img = cv2.imread(_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        bbox = np.array([x, y, w, h])

        return img, bbox  #

    @staticmethod
    def parse_frame_lmdb(f_dict: dict, handles, need_language=False):
        key = f_dict['key']
        f_path = f_dict['path']
        dataset_name = f_dict['dataset']

        handle = handles[dataset_name]

        binfile = handle.get(key.encode('ascii'))
        if binfile is None:
            print("Illegal data detected. %s %s" % (dataset_name, key))
            return None, None
        s = np.frombuffer(binfile, np.uint8)
        if s.size == 0:
            print("Illegal size detected. %s %s" % (dataset_name, key))
            return None, None
        img = cv2.cvtColor(cv2.imdecode(s, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        bbox = f_dict.get('bbox', None)
        if bbox is not None:
            x, y, w, h = bbox
            bbox = np.array([x, y, w, h])

        if not need_language:
            return img, bbox  # [x y w h]
        else:
            lang = f_dict.get('language', '')
            return img, bbox, lang  # [x y w h]

    @staticmethod
    def parse_frame_lmdb_mask(f_dict: dict, handles):
        key = f_dict['key']
        x, y, w, h = f_dict['bbox']
        dataset_name = f_dict['dataset']

        handle = handles[dataset_name]

        binfile = handle.get(key.encode('ascii'))
        if binfile is None:
            print("Illegal data detected. %s %s" % (dataset_name, key))
            return None, None
        s = np.frombuffer(binfile, np.uint8)
        if s.size == 0:
            print("Illegal size detected. %s %s" % (dataset_name, key))
            return None, None
        img = cv2.cvtColor(cv2.imdecode(s, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        bbox = np.array([x, y, w, h])

        segm = f_dict.get('mask', None)
        if segm is not None:
            if isinstance(segm, np.ndarray):
                mask = segm

            elif isinstance(segm, list):
                # polygon -- a single object might consist of multiple parts
                # we merge all parts into one mask rle code
                rles = mask_utils.frPyObjects(segm, img.shape[0], img.shape[1])
                rle = mask_utils.merge(rles)

                mask = mask_utils.decode(rle)

            elif isinstance(segm["counts"], list):
                uncomp_rle = segm
                # convert uncompressed RLE to compressed RLE
                rle = mask_utils.frPyObjects(uncomp_rle, img.shape[0], img.shape[1])

                mask = mask_utils.decode(rle)

            elif isinstance(segm["counts"], str):
                # compressed RLE
                rle = {"counts": segm["counts"], "size": segm["size"]}
                rle["counts"] = rle["counts"].encode()  # str -> bytes
                mask = mask_utils.decode(rle)

            elif isinstance(segm["counts"], bytes):
                # compressed RLE
                rle = {"counts": segm["counts"], "size": segm["size"]}
                mask = mask_utils.decode(rle)
            else:
                raise NotImplementedError
        else:
            mask = None

        lang = f_dict.get('language', '')
        return img, bbox, lang, mask  # [x y w h]

    @staticmethod
    def square_img(img, box, out_sz):  # [x y w h]
        sh, sw = img.shape[:2]
        if sh >= sw:
            scale_f = out_sz / sh
            rh, rw = out_sz, int(sw * scale_f)
            l = (out_sz - rw)
            t = 0
            if l > 0:
                l = np.random.randint(0, l)
        else:
            scale_f = out_sz / sw
            rh, rw = int(sh * scale_f), out_sz
            l = 0
            t = (out_sz - rh)
            if t > 0:
                t = np.random.randint(0, t)

        tmp_img = cv2.resize(img, (rw, rh))
        x, y, w, h = box * scale_f
        out_box = np.array([x + l, y + t, w, h])
        img = np.ones([out_sz, out_sz, 3]) * np.mean(tmp_img, axis=(0, 1))
        img[t:t + rh, l:l + rw] = tmp_img

        return img, out_box

    def crop_patch(self, im, box, out_size, scale_factor, jitter_f, mask=None, returen_mean=False):
        _, j_center, j_size = self.box_jitter(box, jitter_f)

        w_z = j_size[0] + (scale_factor - 1) * np.mean(j_size)
        h_z = j_size[1] + (scale_factor - 1) * np.mean(j_size)
        crop_sz = np.ceil(np.sqrt(w_z * h_z))

        x1 = j_center[0] - crop_sz / 2
        y1 = j_center[1] - crop_sz / 2
        x2 = x1 + crop_sz - 1
        y2 = y1 + crop_sz - 1

        a = out_size / crop_sz
        b = out_size / crop_sz
        c = -a * x1
        d = -b * y1

        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)

        img_mean = np.mean(im, axis=(0, 1))
        patch = cv2.warpAffine(im, mapping,
                               (out_size, out_size),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=img_mean)

        if mask is None:
            x, y, w, h = box
            out_box = np.array([x, y, x + w - 1, y + h - 1])

            out_box[0::2] = out_box[0::2] * a + c
            out_box[1::2] = out_box[1::2] * b + d

            out_box[0::2] = np.clip(out_box[0::2], 0, out_size - 1)
            out_box[1::2] = np.clip(out_box[1::2], 0, out_size - 1)

            out_box[2:] = out_box[2:] - out_box[:2] + 1

            if returen_mean:
                return patch, -np.ones((out_size, out_size)), out_box, img_mean
            else:
                return patch, -np.ones((out_size, out_size)), out_box

        else:
            patch_mask = cv2.warpAffine(mask, mapping,
                                        (out_size, out_size),
                                        cv2.INTER_NEAREST,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=0)
            patch_mask = (patch_mask > 0).astype(np.uint8)

            x, y, w, h = box
            out_box = np.array([x, y, x + w - 1, y + h - 1])

            out_box[0::2] = out_box[0::2] * a + c
            out_box[1::2] = out_box[1::2] * b + d

            out_box[0::2] = np.clip(out_box[0::2], 0, out_size - 1)
            out_box[1::2] = np.clip(out_box[1::2], 0, out_size - 1)

            out_box[2:] = out_box[2:] - out_box[:2] + 1

            return patch, patch_mask, out_box

    def crop_patch_fast(self, im, box, out_size, scale_factor, jitter_f, mask=None):
        _, j_center, j_size = self.box_jitter(box, jitter_f)

        w_z = j_size[0] + (scale_factor - 1) * np.mean(j_size)
        h_z = j_size[1] + (scale_factor - 1) * np.mean(j_size)
        crop_sz = np.ceil(np.sqrt(w_z * h_z))

        x1 = j_center[0] - crop_sz / 2
        y1 = j_center[1] - crop_sz / 2
        x2 = x1 + crop_sz - 1
        y2 = y1 + crop_sz - 1

        a = out_size[1] / crop_sz
        b = out_size[0] / crop_sz
        c = -a * x1
        d = -b * y1

        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)

        patch = cv2.warpAffine(im, mapping,
                               (out_size[1], out_size[0]),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=np.array([103.53, 116.28, 123.675]))  # imagenet BGR

        if mask is None:
            x, y, w, h = box
            out_box = np.array([x, y, x + w - 1, y + h - 1])

            out_box[0::2] = out_box[0::2] * a + c
            out_box[1::2] = out_box[1::2] * b + d

            out_box[0::2] = np.clip(out_box[0::2], 0, out_size[1] - 1)
            out_box[1::2] = np.clip(out_box[1::2], 0, out_size[0] - 1)

            out_box[2:] = out_box[2:] - out_box[:2] + 1

            return patch, -np.ones((out_size[1], out_size[0])), out_box

        else:
            patch_mask = cv2.warpAffine(mask, mapping,
                                        (out_size[1], out_size[0]),
                                        cv2.INTER_NEAREST,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=0)
            patch_mask = (patch_mask > 0).astype(np.uint8)

            x, y, w, h = box
            out_box = np.array([x, y, x + w - 1, y + h - 1])

            out_box[0::2] = out_box[0::2] * a + c
            out_box[1::2] = out_box[1::2] * b + d

            out_box[0::2] = np.clip(out_box[0::2], 0, out_size[1] - 1)
            out_box[1::2] = np.clip(out_box[1::2], 0, out_size[0] - 1)

            out_box[2:] = out_box[2:] - out_box[:2] + 1

            return patch, patch_mask, out_box

    def crop_square_fast(self, im, box, out_size, scale_factor, jitter_f, mask=None):
        _, j_center, j_size = self.box_jitter(box, jitter_f)

        # w_z = j_size[0] + (scale_factor - 1) * np.mean(j_size)
        # h_z = j_size[1] + (scale_factor - 1) * np.mean(j_size)
        w_z = j_size[0] * scale_factor
        h_z = j_size[1] * scale_factor
        crop_sz = [w_z, h_z]

        x1 = j_center[0] - crop_sz[0] / 2
        y1 = j_center[1] - crop_sz[1] / 2
        x2 = x1 + crop_sz[0] - 1
        y2 = y1 + crop_sz[1] - 1

        a = out_size[1] / crop_sz[0]
        b = out_size[0] / crop_sz[1]
        c = -a * x1
        d = -b * y1

        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)

        patch = cv2.warpAffine(im, mapping,
                               (out_size[1], out_size[0]),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=np.array([103.53, 116.28, 123.675]))  # imagenet BGR

        if mask is None:
            x, y, w, h = box
            out_box = np.array([x, y, x + w - 1, y + h - 1])

            out_box[0::2] = out_box[0::2] * a + c
            out_box[1::2] = out_box[1::2] * b + d

            out_box[0::2] = np.clip(out_box[0::2], 0, out_size[1] - 1)
            out_box[1::2] = np.clip(out_box[1::2], 0, out_size[0] - 1)

            out_box[2:] = out_box[2:] - out_box[:2] + 1

            return patch, -np.ones((out_size[1], out_size[0])), out_box

        else:
            patch_mask = cv2.warpAffine(mask, mapping,
                                        (out_size[1], out_size[0]),
                                        cv2.INTER_NEAREST,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=0)
            patch_mask = (patch_mask > 0).astype(np.uint8)

            x, y, w, h = box
            out_box = np.array([x, y, x + w - 1, y + h - 1])

            out_box[0::2] = out_box[0::2] * a + c
            out_box[1::2] = out_box[1::2] * b + d

            out_box[0::2] = np.clip(out_box[0::2], 0, out_size[1] - 1)
            out_box[1::2] = np.clip(out_box[1::2], 0, out_size[0] - 1)

            out_box[2:] = out_box[2:] - out_box[:2] + 1

            return patch, patch_mask, out_box

    def crop_coco(self, im, box, out_size, mask=None):
        out_h, out_w = out_size
        imh, imw = im.shape[:2]

        factor = max(imh / out_h, imw / out_w)
        if factor < 1:
            a, b = 1 / factor, 1 / factor
        else:
            a, b = factor, factor

        resized_h, resized_w = imh * a, imw * b
        c, d = (out_w - resized_w) / 2, (out_h - resized_h) / 2

        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)

        patch = cv2.warpAffine(im, mapping,
                               (out_w, out_h),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=np.mean(im, axis=(0, 1)))

        if mask is None:
            x, y, w, h = box
            out_box = np.array([x, y, x + w - 1, y + h - 1])

            out_box[0::2] = out_box[0::2] * a + c
            out_box[1::2] = out_box[1::2] * b + d

            out_box[0::2] = np.clip(out_box[0::2], 0, out_w - 1)
            out_box[1::2] = np.clip(out_box[1::2], 0, out_h - 1)

            out_box[2:] = out_box[2:] - out_box[:2] + 1

            return patch, -np.ones((out_h, out_w)), out_box

        else:
            patch_mask = cv2.warpAffine(mask, mapping,
                                        (out_w, out_h),
                                        cv2.INTER_NEAREST,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=0)
            patch_mask = (patch_mask > 0).astype(np.uint8)

            x, y, w, h = box
            out_box = np.array([x, y, x + w - 1, y + h - 1])

            out_box[0::2] = out_box[0::2] * a + c
            out_box[1::2] = out_box[1::2] * b + d

            out_box[0::2] = np.clip(out_box[0::2], 0, out_w - 1)
            out_box[1::2] = np.clip(out_box[1::2], 0, out_h - 1)

            out_box[2:] = out_box[2:] - out_box[:2] + 1

            return patch, patch_mask, out_box

    @staticmethod
    def box_jitter(box, jitter_f):
        scale_jitter_f, center_jitter_f = jitter_f

        j_size = box[2:4] * np.exp(np.random.randn(2) * scale_jitter_f)
        max_offset = j_size.mean() * center_jitter_f
        j_center = box[0:2] + 0.5 * box[2:4] + max_offset * (np.random.rand(2) - 0.5)

        j_box = np.concatenate((j_center - j_size * 0.5, j_size))

        return j_box, j_center, j_size

    # @staticmethod
    # def gaussian_blur(im):
    #     max_kernel = 7
    #     max_kernel = ((max_kernel + 1) // 2)
    #     kernel_size = (
    #         np.random.randint(1, max_kernel) * 2 + 1,
    #         np.random.randint(1, max_kernel) * 2 + 1,
    #     )
    #     im = cv2.GaussianBlur(im, kernel_size, 0)
    #     return im
    #
    # @staticmethod
    # def color_dropout(im):
    #     c = np.random.randint(0, 3)
    #     im[:, :, c] = np.clip(im[:, :, c] * np.random.uniform(0.2, 0.5), 0, 255).astype(np.uint8)
    #     return im
    #
    # @staticmethod
    # def color_jitter(im, jitter_f=0.2):
    #     im = np.clip(
    #         im * np.random.uniform(np.maximum(0, 1 - jitter_f),
    #                                1 + jitter_f,
    #                                [1, 1, 3]),
    #         0, 255).astype(np.uint8)
    #     return im
    #
    # @staticmethod
    # def color_gray(im):
    #     img_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    #     im = np.stack([img_gray, img_gray, img_gray], axis=2)
    #     return im

    @staticmethod
    def horizontal_flip(image, bbox=None):  # [x, y, w, h]
        image = cv2.flip(image, 1)
        width = image.shape[1]

        if bbox is None:
            return image
        else:
            x1, y1, w, h = bbox
            x2 = x1 + w - 1
            y2 = y1 + h - 1

            bbox = np.array([width - 1 - x2, y1, width - 1 - x1, y2])
            bbox[2:] = bbox[2:] - bbox[:2] + 1

            return image, bbox
