import os
import sys
from typing import Callable, Optional

import lmdb
import numpy as np
from PIL import Image

from lib.dataset import SubSet


class LMDBData(object):
    def __init__(self, save_dir):
        try:
            os.makedirs(save_dir, exist_ok=False)
        except FileExistsError:
            print(f'Directory had been made: {save_dir}. Please check whether the lmdb had been built.')
            sys.exit()

        self.db = lmdb.open(save_dir, map_size=1099511627776, readonly=False,
                            meminit=False, map_async=True)  # max(B) 1TB
        self.txn = self.db.begin(write=True)

    def put_image(self, img_file, key: str):
        has = self.txn.get(key.encode('ascii'))
        if has is not None:
            print('... this frame has been saved')
            return

        with open(img_file, 'rb') as f:
            value = f.read()
        self.txn.put(key.encode('ascii'), value)

    def commit(self):
        self.txn.commit()
        self.txn = self.db.begin(write=True)

    def sync_and_close(self):
        self.db.sync()
        self.db.close()


class DatasetLoader(object):
    def __init__(self):

        self.dataset_name = None
        self.data_dir = None

        self.video_list = None
        self.imw = None
        self.imh = None

        self.video_name = None
        self.img_list = None
        self.gt_list = None
        self.lang_list = None
        self.key_list = None

        self.mask_list = None
        self.object_id = None
        self.category_id = None

    def get_video_info(self, *args, **kwargs):
        raise NotImplementedError

    def get_frame_dict(self, idx, lmdb_put_img_fn: Callable = None):
        frame_dict_list = []
        for im_i, _ in enumerate(self.img_list):

            frame_dict = dict()
            frame_dict['path'] = self.img_list[im_i]

            # for some datasets (e.g. GOT10k),
            # different frames of the same video may have different resolution
            if self.imw is None or self.imh is None:
                img = Image.open(os.path.join(self.data_dir, self.img_list[im_i]))
                self.imw, self.imh = img.size

            if self.gt_list is not None:
                x1, y1, w, h = self.gt_list[im_i]
                x2 = x1 + w - 1
                y2 = y1 + h - 1

                x1, x2 = np.clip([x1, x2], 0, self.imw - 1)
                y1, y2 = np.clip([y1, y2], 0, self.imh - 1)

                w = x2 - x1 + 1
                h = y2 - y1 + 1
                frame_dict['bbox'] = [x1, y1, w, h]

            if self.__getattribute__('key_list') is None:
                frame_dict['key'] = os.path.join(self.video_name,
                                                 self.img_list[im_i].split('/')[-1].split('.')[0])
            else:
                frame_dict['key'] = self.key_list[im_i]

            # some datasets don't have language annotation
            if self.lang_list is not None:
                frame_dict['language'] = self.lang_list[im_i]

            if self.category_id is not None:
                frame_dict['category_id'] = int(self.category_id)

            frame_dict['dataset'] = self.dataset_name
            frame_dict['video'] = self.video_name
            frame_dict['length'] = len(self.img_list)
            frame_dict['size'] = [int(self.imw), int(self.imh)]

            if self.mask_list is not None:
                frame_dict['mask'] = self.mask_list[im_i]

            frame_dict_list.append(frame_dict)

            if lmdb_put_img_fn is not None:
                lmdb_put_img_fn(img_file=os.path.join(self.data_dir, self.img_list[im_i]),
                                key=frame_dict['key'])
                print('process ... {} / {}, {} / {}, save json, save lmdb'.format(
                    idx, len(self.video_list), frame_dict['path'], frame_dict['length']))
            else:
                print('process ... {} / {}, {} / {}, save json'.format(
                    idx, len(self.video_list), frame_dict['path'], frame_dict['length']))

        return frame_dict_list


def parse_loop(name, save_dir, data_set: DatasetLoader, json_obj: SubSet, lmdb_obj: Optional[LMDBData]):

    video_num = 0
    frame_num = 0
    for v_i, v_name in enumerate(data_set.video_list):
        data_set.get_video_info(v_name)
        if lmdb_obj is not None:
            frame_dict_list = data_set.get_frame_dict(idx=v_i, lmdb_put_img_fn=lmdb_obj.put_image)
        else:
            frame_dict_list = data_set.get_frame_dict(idx=v_i, lmdb_put_img_fn=None)
        if len(frame_dict_list) > 0:
            json_obj.append(frame_dict_list)
            if lmdb_obj is not None:
                lmdb_obj.commit()
            video_num += 1
            frame_num += len(frame_dict_list)
        else:
            pass

    print('#################################')
    print('[{}] -- {:d} videos, {:d} frames, done !'.format(name, video_num, frame_num))
    print('#################################')
    if lmdb_obj is not None:
        lmdb_obj.sync_and_close()
    json_obj.save(save_dir)
