import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List


strList = List[str]


def load_votlt(root):
    video_name_list = np.loadtxt(os.path.join(root, 'list.txt'), dtype=np.str)

    if len(video_name_list) > 35:
        name = 'VOT19-LT'
    else:
        name = 'VOT18-LT'

    video_list = []
    for v_n in tqdm(video_name_list, ascii=True, desc='loading %s' % name):
        gt_path = os.path.join(root, v_n, 'groundtruth.txt')
        im_dir = os.path.join(root, v_n, 'color')
        lang_path = os.path.join(root, 'RefLTB50', v_n, 'language.txt')

        im_list: strList = os.listdir(im_dir)
        im_list = sorted(im_list)

        gts = np.loadtxt(gt_path, delimiter=',').tolist()  # [x y w h]
        ims = [os.path.join(im_dir, im_f) for im_f in im_list if 'jpg' in im_f]
        with open(lang_path, 'r') as f:
            language = f.readlines()

        language = [f.strip().lower() for f in language]

        video_list.append([v_n, ims, gts, language])

    return video_list


def load_votlt22(root):
    video_name_list = np.loadtxt(os.path.join(root, 'list.txt'), dtype=np.str)

    name = 'VOT22-LT'

    video_list = []
    for v_n in tqdm(video_name_list, ascii=True, desc='loading %s' % name):
        gt_path = os.path.join(root, v_n, 'groundtruth.txt')
        im_dir = os.path.join(root, v_n, 'color')

        im_list: strList = os.listdir(im_dir)
        im_list = sorted(im_list)

        # gts = np.loadtxt(gt_path, delimiter=',').tolist()  # [x y w h]

        with open(gt_path, 'r') as f:
            tmp = f.readline()
        gts = [float(v) for v in tmp.split(',')]

        ims = [os.path.join(im_dir, im_f) for im_f in im_list if 'jpg' in im_f]

        video_list.append([v_n, ims, [gts], ''])

    return video_list


def load_lasot(root):
    video_name_list = np.loadtxt(os.path.join(root, '..', 'testing_set.txt'), dtype=np.str).tolist()

    video_list = []
    for v_n in tqdm(video_name_list, ascii=True, desc='loading LaSOT'):
        gt_path = os.path.join(root, v_n, 'groundtruth.txt')
        lang_path = os.path.join(root, v_n, 'nlp.txt')
        im_dir = os.path.join(root, v_n, 'img')

        im_list: strList = os.listdir(im_dir)
        im_list = sorted(im_list)

        gts = np.loadtxt(gt_path, delimiter=',').tolist()  # [x y w h]
        with open(lang_path, 'r') as f:
            lang = f.readline().lower()
        ims = [os.path.join(im_dir, im_f) for im_f in im_list if 'jpg' in im_f]

        video_list.append([v_n, ims, gts, lang.strip()])

    return video_list


def load_got10k(root):
    video_name_list = os.listdir(root)
    video_name_list = [v for v in video_name_list if os.path.isdir(os.path.join(root, v))]
    video_name_list = sorted(video_name_list)

    video_list = []
    for v_n in tqdm(video_name_list, ascii=True, desc='loading GOT10k'):
        gt_path = os.path.join(root, v_n, 'groundtruth.txt')
        im_dir = os.path.join(root, v_n)

        im_list: strList = os.listdir(im_dir)
        im_list = sorted(im_list)
        im_list = [im_f for im_f in im_list if 'jpg' in im_f or 'png' in im_f]

        gts = np.loadtxt(gt_path, delimiter=',').reshape(-1, 4).tolist()  # [x y w h]
        ims = [os.path.join(im_dir, im_f) for im_f in im_list]

        video_list.append([v_n, ims, gts, ''])

    return video_list


def load_tnl2k(root):
    video_name_list = os.listdir(root)
    video_name_list = [v for v in video_name_list if os.path.isdir(os.path.join(root, v))]
    video_name_list = sorted(video_name_list)

    video_list = []
    for v_n in tqdm(video_name_list, ascii=True, desc='loading TNL2K'):
        gt_path = os.path.join(root, v_n, 'groundtruth.txt')
        lang_path = os.path.join(root, v_n, 'language.txt')
        im_dir = os.path.join(root, v_n, 'imgs')

        im_list: strList = os.listdir(im_dir)
        im_list = sorted(im_list)
        im_list = [im_f for im_f in im_list if 'jpg' in im_f or 'png' in im_f]
        # test_015_Sord_video_Q01_done/imgs/00000492.pn_error

        gts = np.loadtxt(gt_path, delimiter=',').reshape(-1, 4).tolist()  # [x y w h]
        with open(lang_path, 'r') as f:
            lang = f.readline().lower()
            if len(lang) == 0:  # Fight_video_6-Done
                lang = '[MASK]'

        lang = lang.replace('we want to track ', '').strip()
        lang = ('+' + lang).replace('+the ', '').replace('+a ', '').strip('+')

        ims = [os.path.join(im_dir, im_f) for im_f in im_list]

        video_list.append([v_n, ims, gts, lang.strip()])

    return video_list


def load_itb(root):
    video_name_list = os.listdir(root)
    video_name_list = [v for v in video_name_list if os.path.isdir(os.path.join(root, v))]
    video_name_list = sorted(video_name_list)

    sub_video_name_list = []
    for d in video_name_list:
        tmp_list = os.listdir(os.path.join(root, d))
        sub_video_name_list += [os.path.join(d, v) for v in tmp_list if os.path.isdir(os.path.join(root, d, v))]

    video_list = []
    for v_n in tqdm(sub_video_name_list, ascii=True, desc='loading ITB'):
        gt_path = os.path.join(root, v_n, 'groundtruth.txt')
        im_dir = os.path.join(root, v_n)

        im_list: strList = os.listdir(im_dir)
        im_list = sorted(im_list)
        im_list = [im_f for im_f in im_list if 'jpg' in im_f or 'png' in im_f]

        gts = np.loadtxt(gt_path, delimiter=',').reshape(-1, 4).tolist()  # [x y w h]

        ims = [os.path.join(im_dir, im_f) for im_f in im_list]

        video_list.append([v_n.split('/')[-1], ims, gts, ''])

    return video_list


def load_trackingnet(root):
    video_root = os.path.join(root, 'frames')
    annos_root = os.path.join(root, 'anno')

    video_name_list = os.listdir(video_root)
    video_name_list = [v for v in video_name_list if os.path.isdir(os.path.join(video_root, v))]
    video_name_list = sorted(video_name_list)

    video_list = []
    for v_n in tqdm(video_name_list, ascii=True, desc='loading TrackingNet'):
        gt_path = os.path.join(annos_root, v_n + '.txt')
        im_dir = os.path.join(video_root, v_n)

        im_list = os.listdir(im_dir)
        im_list = sorted(im_list)

        ids = []
        for f in im_list:
            ids.append(int(f.split('.')[0]))
        im_list = np.array(im_list)
        im_list = im_list[np.argsort(np.array(ids))].tolist()

        im_list = [im_f for im_f in im_list if 'jpg' in im_f or 'png' in im_f]
        # test_015_Sord_video_Q01_done/imgs/00000492.pn_error

        gts = np.loadtxt(gt_path, delimiter=',').reshape(-1, 4).tolist()  # [x y w h]
        ims = [os.path.join(im_dir, im_f) for im_f in im_list]

        video_list.append([v_n, ims, gts, ''])

    return video_list


def load_otb_lang(root):
    video_root = os.path.join(root, 'OTB_videos')
    annos_root = os.path.join(root, 'OTB_videos')
    lang_root = os.path.join(root, 'OTB_query_test')

    video_name_list = os.listdir(os.path.join(root, 'OTB_query_test'))
    video_name_list = [v.replace('.txt', '') for v in video_name_list if '.txt' in v]
    video_name_list = [v for v in video_name_list if os.path.isdir(os.path.join(video_root, v))]
    video_name_list = sorted(video_name_list)

    video_list = []
    for v_n in tqdm(video_name_list, ascii=True, desc='loading OTB-LANG'):
        gt_path = os.path.join(annos_root, v_n, 'groundtruth_rect.txt')
        im_dir = os.path.join(video_root, v_n, 'img')
        lang_path = os.path.join(lang_root, v_n + '.txt')

        im_list = os.listdir(im_dir)
        im_list = sorted(im_list)

        ids = []
        for f in im_list:
            ids.append(int(f.split('.')[0]))
        im_list = np.array(im_list)
        im_list = im_list[np.argsort(np.array(ids))].tolist()

        im_list = [im_f for im_f in im_list if 'jpg' in im_f or 'png' in im_f]
        ims = [os.path.join(im_dir, im_f) for im_f in im_list]

        try:
            gts = np.loadtxt(gt_path, delimiter=',').reshape(-1, 4).tolist()  # [x y w h]
        except:
            gts = np.loadtxt(gt_path, delimiter='\t').reshape(-1, 4).tolist()  # [x y w h]

        with open(lang_path, 'r') as f:
            lang = f.readline().lower()

        video_list.append([v_n, ims, gts, lang.strip()])

    return video_list


def load_otb(root):
    video_root = root
    annos_root = root

    video_name_list = os.listdir(root)
    video_name_list = [v for v in video_name_list if os.path.isdir(os.path.join(video_root, v))]
    video_name_list = sorted(video_name_list)

    sequence_info_list = [
        {"name": "Basketball", "path": "Basketball/img", "startFrame": 1, "endFrame": 725, "nz": 4, "ext": "jpg",
         "anno_path": "Basketball/groundtruth_rect.txt",
         "object_class": "person"},
        {"name": "Biker", "path": "Biker/img", "startFrame": 1, "endFrame": 142, "nz": 4, "ext": "jpg",
         "anno_path": "Biker/groundtruth_rect.txt",
         "object_class": "person head"},
        {"name": "Bird1", "path": "Bird1/img", "startFrame": 1, "endFrame": 408, "nz": 4, "ext": "jpg",
         "anno_path": "Bird1/groundtruth_rect.txt",
         "object_class": "bird"},
        {"name": "Bird2", "path": "Bird2/img", "startFrame": 1, "endFrame": 99, "nz": 4, "ext": "jpg",
         "anno_path": "Bird2/groundtruth_rect.txt",
         "object_class": "bird"},
        {"name": "BlurBody", "path": "BlurBody/img", "startFrame": 1, "endFrame": 334, "nz": 4, "ext": "jpg",
         "anno_path": "BlurBody/groundtruth_rect.txt",
         "object_class": "person"},
        {"name": "BlurCar1", "path": "BlurCar1/img", "startFrame": 247, "endFrame": 988, "nz": 4, "ext": "jpg",
         "anno_path": "BlurCar1/groundtruth_rect.txt",
         "object_class": "car"},
        {"name": "BlurCar2", "path": "BlurCar2/img", "startFrame": 1, "endFrame": 585, "nz": 4, "ext": "jpg",
         "anno_path": "BlurCar2/groundtruth_rect.txt",
         "object_class": "car"},
        {"name": "BlurCar3", "path": "BlurCar3/img", "startFrame": 3, "endFrame": 359, "nz": 4, "ext": "jpg",
         "anno_path": "BlurCar3/groundtruth_rect.txt",
         "object_class": "car"},
        {"name": "BlurCar4", "path": "BlurCar4/img", "startFrame": 18, "endFrame": 397, "nz": 4, "ext": "jpg",
         "anno_path": "BlurCar4/groundtruth_rect.txt",
         "object_class": "car"},
        {"name": "BlurFace", "path": "BlurFace/img", "startFrame": 1, "endFrame": 493, "nz": 4, "ext": "jpg",
         "anno_path": "BlurFace/groundtruth_rect.txt",
         "object_class": "face"},
        {"name": "BlurOwl", "path": "BlurOwl/img", "startFrame": 1, "endFrame": 631, "nz": 4, "ext": "jpg",
         "anno_path": "BlurOwl/groundtruth_rect.txt",
         "object_class": "other"},
        {"name": "Board", "path": "Board/img", "startFrame": 1, "endFrame": 698, "nz": 5, "ext": "jpg",
         "anno_path": "Board/groundtruth_rect.txt",
         "object_class": "other"},
        {"name": "Bolt", "path": "Bolt/img", "startFrame": 1, "endFrame": 350, "nz": 4, "ext": "jpg",
         "anno_path": "Bolt/groundtruth_rect.txt",
         "object_class": "person"},
        {"name": "Bolt2", "path": "Bolt2/img", "startFrame": 1, "endFrame": 293, "nz": 4, "ext": "jpg",
         "anno_path": "Bolt2/groundtruth_rect.txt",
         "object_class": "person"},
        {"name": "Box", "path": "Box/img", "startFrame": 1, "endFrame": 1161, "nz": 4, "ext": "jpg",
         "anno_path": "Box/groundtruth_rect.txt",
         "object_class": "other"},
        {"name": "Boy", "path": "Boy/img", "startFrame": 1, "endFrame": 602, "nz": 4, "ext": "jpg",
         "anno_path": "Boy/groundtruth_rect.txt",
         "object_class": "face"},
        {"name": "Car1", "path": "Car1/img", "startFrame": 1, "endFrame": 1020, "nz": 4, "ext": "jpg",
         "anno_path": "Car1/groundtruth_rect.txt",
         "object_class": "car"},
        {"name": "Car2", "path": "Car2/img", "startFrame": 1, "endFrame": 913, "nz": 4, "ext": "jpg",
         "anno_path": "Car2/groundtruth_rect.txt",
         "object_class": "car"},
        {"name": "Car24", "path": "Car24/img", "startFrame": 1, "endFrame": 3059, "nz": 4, "ext": "jpg",
         "anno_path": "Car24/groundtruth_rect.txt",
         "object_class": "car"},
        {"name": "Car4", "path": "Car4/img", "startFrame": 1, "endFrame": 659, "nz": 4, "ext": "jpg",
         "anno_path": "Car4/groundtruth_rect.txt",
         "object_class": "car"},
        {"name": "CarDark", "path": "CarDark/img", "startFrame": 1, "endFrame": 393, "nz": 4, "ext": "jpg",
         "anno_path": "CarDark/groundtruth_rect.txt",
         "object_class": "car"},
        {"name": "CarScale", "path": "CarScale/img", "startFrame": 1, "endFrame": 252, "nz": 4, "ext": "jpg",
         "anno_path": "CarScale/groundtruth_rect.txt",
         "object_class": "car"},
        {"name": "ClifBar", "path": "ClifBar/img", "startFrame": 1, "endFrame": 472, "nz": 4, "ext": "jpg",
         "anno_path": "ClifBar/groundtruth_rect.txt",
         "object_class": "other"},
        {"name": "Coke", "path": "Coke/img", "startFrame": 1, "endFrame": 291, "nz": 4, "ext": "jpg",
         "anno_path": "Coke/groundtruth_rect.txt",
         "object_class": "other"},
        {"name": "Couple", "path": "Couple/img", "startFrame": 1, "endFrame": 140, "nz": 4, "ext": "jpg",
         "anno_path": "Couple/groundtruth_rect.txt",
         "object_class": "person"},
        {"name": "Coupon", "path": "Coupon/img", "startFrame": 1, "endFrame": 327, "nz": 4, "ext": "jpg",
         "anno_path": "Coupon/groundtruth_rect.txt",
         "object_class": "other"},
        {"name": "Crossing", "path": "Crossing/img", "startFrame": 1, "endFrame": 120, "nz": 4, "ext": "jpg",
         "anno_path": "Crossing/groundtruth_rect.txt",
         "object_class": "person"},
        {"name": "Crowds", "path": "Crowds/img", "startFrame": 1, "endFrame": 347, "nz": 4, "ext": "jpg",
         "anno_path": "Crowds/groundtruth_rect.txt",
         "object_class": "person"},
        {"name": "Dancer", "path": "Dancer/img", "startFrame": 1, "endFrame": 225, "nz": 4, "ext": "jpg",
         "anno_path": "Dancer/groundtruth_rect.txt",
         "object_class": "person"},
        {"name": "Dancer2", "path": "Dancer2/img", "startFrame": 1, "endFrame": 150, "nz": 4, "ext": "jpg",
         "anno_path": "Dancer2/groundtruth_rect.txt",
         "object_class": "person"},
        {"name": "David", "path": "David/img", "startFrame": 300, "endFrame": 770, "nz": 4, "ext": "jpg",
         "anno_path": "David/groundtruth_rect.txt",
         "object_class": "face"},
        {"name": "David2", "path": "David2/img", "startFrame": 1, "endFrame": 537, "nz": 4, "ext": "jpg",
         "anno_path": "David2/groundtruth_rect.txt",
         "object_class": "face"},
        {"name": "David3", "path": "David3/img", "startFrame": 1, "endFrame": 252, "nz": 4, "ext": "jpg",
         "anno_path": "David3/groundtruth_rect.txt",
         "object_class": "person"},
        {"name": "Deer", "path": "Deer/img", "startFrame": 1, "endFrame": 71, "nz": 4, "ext": "jpg",
         "anno_path": "Deer/groundtruth_rect.txt",
         "object_class": "mammal"},
        {"name": "Diving", "path": "Diving/img", "startFrame": 1, "endFrame": 215, "nz": 4, "ext": "jpg",
         "anno_path": "Diving/groundtruth_rect.txt",
         "object_class": "person"},
        {"name": "Dog", "path": "Dog/img", "startFrame": 1, "endFrame": 127, "nz": 4, "ext": "jpg",
         "anno_path": "Dog/groundtruth_rect.txt",
         "object_class": "dog"},
        {"name": "Dog1", "path": "Dog1/img", "startFrame": 1, "endFrame": 1350, "nz": 4, "ext": "jpg",
         "anno_path": "Dog1/groundtruth_rect.txt",
         "object_class": "dog"},
        {"name": "Doll", "path": "Doll/img", "startFrame": 1, "endFrame": 3872, "nz": 4, "ext": "jpg",
         "anno_path": "Doll/groundtruth_rect.txt",
         "object_class": "other"},
        {"name": "DragonBaby", "path": "DragonBaby/img", "startFrame": 1, "endFrame": 113, "nz": 4, "ext": "jpg",
         "anno_path": "DragonBaby/groundtruth_rect.txt",
         "object_class": "face"},
        {"name": "Dudek", "path": "Dudek/img", "startFrame": 1, "endFrame": 1145, "nz": 4, "ext": "jpg",
         "anno_path": "Dudek/groundtruth_rect.txt",
         "object_class": "face"},
        {"name": "FaceOcc1", "path": "FaceOcc1/img", "startFrame": 1, "endFrame": 892, "nz": 4, "ext": "jpg",
         "anno_path": "FaceOcc1/groundtruth_rect.txt",
         "object_class": "face"},
        {"name": "FaceOcc2", "path": "FaceOcc2/img", "startFrame": 1, "endFrame": 812, "nz": 4, "ext": "jpg",
         "anno_path": "FaceOcc2/groundtruth_rect.txt",
         "object_class": "face"},
        {"name": "Fish", "path": "Fish/img", "startFrame": 1, "endFrame": 476, "nz": 4, "ext": "jpg",
         "anno_path": "Fish/groundtruth_rect.txt",
         "object_class": "other"},
        {"name": "FleetFace", "path": "FleetFace/img", "startFrame": 1, "endFrame": 707, "nz": 4, "ext": "jpg",
         "anno_path": "FleetFace/groundtruth_rect.txt",
         "object_class": "face"},
        {"name": "Football", "path": "Football/img", "startFrame": 1, "endFrame": 362, "nz": 4, "ext": "jpg",
         "anno_path": "Football/groundtruth_rect.txt",
         "object_class": "person head"},
        {"name": "Football1", "path": "Football1/img", "startFrame": 1, "endFrame": 74, "nz": 4, "ext": "jpg",
         "anno_path": "Football1/groundtruth_rect.txt",
         "object_class": "face"},
        {"name": "Freeman1", "path": "Freeman1/img", "startFrame": 1, "endFrame": 326, "nz": 4, "ext": "jpg",
         "anno_path": "Freeman1/groundtruth_rect.txt",
         "object_class": "face"},
        {"name": "Freeman3", "path": "Freeman3/img", "startFrame": 1, "endFrame": 460, "nz": 4, "ext": "jpg",
         "anno_path": "Freeman3/groundtruth_rect.txt",
         "object_class": "face"},
        {"name": "Freeman4", "path": "Freeman4/img", "startFrame": 1, "endFrame": 283, "nz": 4, "ext": "jpg",
         "anno_path": "Freeman4/groundtruth_rect.txt",
         "object_class": "face"},
        {"name": "Girl", "path": "Girl/img", "startFrame": 1, "endFrame": 500, "nz": 4, "ext": "jpg",
         "anno_path": "Girl/groundtruth_rect.txt",
         "object_class": "face"},
        {"name": "Girl2", "path": "Girl2/img", "startFrame": 1, "endFrame": 1500, "nz": 4, "ext": "jpg",
         "anno_path": "Girl2/groundtruth_rect.txt",
         "object_class": "person"},
        {"name": "Gym", "path": "Gym/img", "startFrame": 1, "endFrame": 767, "nz": 4, "ext": "jpg",
         "anno_path": "Gym/groundtruth_rect.txt",
         "object_class": "person"},
        {"name": "Human2", "path": "Human2/img", "startFrame": 1, "endFrame": 1128, "nz": 4, "ext": "jpg",
         "anno_path": "Human2/groundtruth_rect.txt",
         "object_class": "person"},
        {"name": "Human3", "path": "Human3/img", "startFrame": 1, "endFrame": 1698, "nz": 4, "ext": "jpg",
         "anno_path": "Human3/groundtruth_rect.txt",
         "object_class": "person"},
        {"name": "Human4_2", "path": "Human4/img", "startFrame": 1, "endFrame": 667, "nz": 4, "ext": "jpg",
         "anno_path": "Human4/groundtruth_rect.2.txt",
         "object_class": "person"},
        {"name": "Human5", "path": "Human5/img", "startFrame": 1, "endFrame": 713, "nz": 4, "ext": "jpg",
         "anno_path": "Human5/groundtruth_rect.txt",
         "object_class": "person"},
        {"name": "Human6", "path": "Human6/img", "startFrame": 1, "endFrame": 792, "nz": 4, "ext": "jpg",
         "anno_path": "Human6/groundtruth_rect.txt",
         "object_class": "person"},
        {"name": "Human7", "path": "Human7/img", "startFrame": 1, "endFrame": 250, "nz": 4, "ext": "jpg",
         "anno_path": "Human7/groundtruth_rect.txt",
         "object_class": "person"},
        {"name": "Human8", "path": "Human8/img", "startFrame": 1, "endFrame": 128, "nz": 4, "ext": "jpg",
         "anno_path": "Human8/groundtruth_rect.txt",
         "object_class": "person"},
        {"name": "Human9", "path": "Human9/img", "startFrame": 1, "endFrame": 305, "nz": 4, "ext": "jpg",
         "anno_path": "Human9/groundtruth_rect.txt",
         "object_class": "person"},
        {"name": "Ironman", "path": "Ironman/img", "startFrame": 1, "endFrame": 166, "nz": 4, "ext": "jpg",
         "anno_path": "Ironman/groundtruth_rect.txt",
         "object_class": "person head"},
        {"name": "Jogging_1", "path": "Jogging/img", "startFrame": 1, "endFrame": 307, "nz": 4, "ext": "jpg",
         "anno_path": "Jogging/groundtruth_rect.1.txt",
         "object_class": "person"},
        {"name": "Jogging_2", "path": "Jogging/img", "startFrame": 1, "endFrame": 307, "nz": 4, "ext": "jpg",
         "anno_path": "Jogging/groundtruth_rect.2.txt",
         "object_class": "person"},
        {"name": "Jump", "path": "Jump/img", "startFrame": 1, "endFrame": 122, "nz": 4, "ext": "jpg",
         "anno_path": "Jump/groundtruth_rect.txt",
         "object_class": "person"},
        {"name": "Jumping", "path": "Jumping/img", "startFrame": 1, "endFrame": 313, "nz": 4, "ext": "jpg",
         "anno_path": "Jumping/groundtruth_rect.txt",
         "object_class": "face"},
        {"name": "KiteSurf", "path": "KiteSurf/img", "startFrame": 1, "endFrame": 84, "nz": 4, "ext": "jpg",
         "anno_path": "KiteSurf/groundtruth_rect.txt",
         "object_class": "face"},
        {"name": "Lemming", "path": "Lemming/img", "startFrame": 1, "endFrame": 1336, "nz": 4, "ext": "jpg",
         "anno_path": "Lemming/groundtruth_rect.txt",
         "object_class": "other"},
        {"name": "Liquor", "path": "Liquor/img", "startFrame": 1, "endFrame": 1741, "nz": 4, "ext": "jpg",
         "anno_path": "Liquor/groundtruth_rect.txt",
         "object_class": "other"},
        {"name": "Man", "path": "Man/img", "startFrame": 1, "endFrame": 134, "nz": 4, "ext": "jpg",
         "anno_path": "Man/groundtruth_rect.txt",
         "object_class": "face"},
        {"name": "Matrix", "path": "Matrix/img", "startFrame": 1, "endFrame": 100, "nz": 4, "ext": "jpg",
         "anno_path": "Matrix/groundtruth_rect.txt",
         "object_class": "person head"},
        {"name": "Mhyang", "path": "Mhyang/img", "startFrame": 1, "endFrame": 1490, "nz": 4, "ext": "jpg",
         "anno_path": "Mhyang/groundtruth_rect.txt",
         "object_class": "face"},
        {"name": "MotorRolling", "path": "MotorRolling/img", "startFrame": 1, "endFrame": 164, "nz": 4, "ext": "jpg",
         "anno_path": "MotorRolling/groundtruth_rect.txt",
         "object_class": "vehicle"},
        {"name": "MountainBike", "path": "MountainBike/img", "startFrame": 1, "endFrame": 228, "nz": 4, "ext": "jpg",
         "anno_path": "MountainBike/groundtruth_rect.txt",
         "object_class": "bicycle"},
        {"name": "Panda", "path": "Panda/img", "startFrame": 1, "endFrame": 1000, "nz": 4, "ext": "jpg",
         "anno_path": "Panda/groundtruth_rect.txt",
         "object_class": "mammal"},
        {"name": "RedTeam", "path": "RedTeam/img", "startFrame": 1, "endFrame": 1918, "nz": 4, "ext": "jpg",
         "anno_path": "RedTeam/groundtruth_rect.txt",
         "object_class": "vehicle"},
        {"name": "Rubik", "path": "Rubik/img", "startFrame": 1, "endFrame": 1997, "nz": 4, "ext": "jpg",
         "anno_path": "Rubik/groundtruth_rect.txt",
         "object_class": "other"},
        {"name": "Shaking", "path": "Shaking/img", "startFrame": 1, "endFrame": 365, "nz": 4, "ext": "jpg",
         "anno_path": "Shaking/groundtruth_rect.txt",
         "object_class": "face"},
        {"name": "Singer1", "path": "Singer1/img", "startFrame": 1, "endFrame": 351, "nz": 4, "ext": "jpg",
         "anno_path": "Singer1/groundtruth_rect.txt",
         "object_class": "person"},
        {"name": "Singer2", "path": "Singer2/img", "startFrame": 1, "endFrame": 366, "nz": 4, "ext": "jpg",
         "anno_path": "Singer2/groundtruth_rect.txt",
         "object_class": "person"},
        {"name": "Skater", "path": "Skater/img", "startFrame": 1, "endFrame": 160, "nz": 4, "ext": "jpg",
         "anno_path": "Skater/groundtruth_rect.txt",
         "object_class": "person"},
        {"name": "Skater2", "path": "Skater2/img", "startFrame": 1, "endFrame": 435, "nz": 4, "ext": "jpg",
         "anno_path": "Skater2/groundtruth_rect.txt",
         "object_class": "person"},
        {"name": "Skating1", "path": "Skating1/img", "startFrame": 1, "endFrame": 400, "nz": 4, "ext": "jpg",
         "anno_path": "Skating1/groundtruth_rect.txt",
         "object_class": "person"},
        {"name": "Skating2_1", "path": "Skating2/img", "startFrame": 1, "endFrame": 473, "nz": 4, "ext": "jpg",
         "anno_path": "Skating2/groundtruth_rect.1.txt",
         "object_class": "person"},
        {"name": "Skating2_2", "path": "Skating2/img", "startFrame": 1, "endFrame": 473, "nz": 4, "ext": "jpg",
         "anno_path": "Skating2/groundtruth_rect.2.txt",
         "object_class": "person"},
        {"name": "Skiing", "path": "Skiing/img", "startFrame": 1, "endFrame": 81, "nz": 4, "ext": "jpg",
         "anno_path": "Skiing/groundtruth_rect.txt",
         "object_class": "person"},
        {"name": "Soccer", "path": "Soccer/img", "startFrame": 1, "endFrame": 392, "nz": 4, "ext": "jpg",
         "anno_path": "Soccer/groundtruth_rect.txt",
         "object_class": "face"},
        {"name": "Subway", "path": "Subway/img", "startFrame": 1, "endFrame": 175, "nz": 4, "ext": "jpg",
         "anno_path": "Subway/groundtruth_rect.txt",
         "object_class": "person"},
        {"name": "Surfer", "path": "Surfer/img", "startFrame": 1, "endFrame": 376, "nz": 4, "ext": "jpg",
         "anno_path": "Surfer/groundtruth_rect.txt",
         "object_class": "person head"},
        {"name": "Suv", "path": "Suv/img", "startFrame": 1, "endFrame": 945, "nz": 4, "ext": "jpg",
         "anno_path": "Suv/groundtruth_rect.txt",
         "object_class": "car"},
        {"name": "Sylvester", "path": "Sylvester/img", "startFrame": 1, "endFrame": 1345, "nz": 4, "ext": "jpg",
         "anno_path": "Sylvester/groundtruth_rect.txt",
         "object_class": "other"},
        {"name": "Tiger1", "path": "Tiger1/img", "startFrame": 1, "endFrame": 354, "nz": 4, "ext": "jpg",
         "anno_path": "Tiger1/groundtruth_rect.txt", "initOmit": 5,
         "object_class": "other"},
        {"name": "Tiger2", "path": "Tiger2/img", "startFrame": 1, "endFrame": 365, "nz": 4, "ext": "jpg",
         "anno_path": "Tiger2/groundtruth_rect.txt",
         "object_class": "other"},
        {"name": "Toy", "path": "Toy/img", "startFrame": 1, "endFrame": 271, "nz": 4, "ext": "jpg",
         "anno_path": "Toy/groundtruth_rect.txt",
         "object_class": "other"},
        {"name": "Trans", "path": "Trans/img", "startFrame": 1, "endFrame": 124, "nz": 4, "ext": "jpg",
         "anno_path": "Trans/groundtruth_rect.txt",
         "object_class": "other"},
        {"name": "Trellis", "path": "Trellis/img", "startFrame": 1, "endFrame": 569, "nz": 4, "ext": "jpg",
         "anno_path": "Trellis/groundtruth_rect.txt",
         "object_class": "face"},
        {"name": "Twinnings", "path": "Twinnings/img", "startFrame": 1, "endFrame": 472, "nz": 4, "ext": "jpg",
         "anno_path": "Twinnings/groundtruth_rect.txt",
         "object_class": "other"},
        {"name": "Vase", "path": "Vase/img", "startFrame": 1, "endFrame": 271, "nz": 4, "ext": "jpg",
         "anno_path": "Vase/groundtruth_rect.txt",
         "object_class": "other"},
        {"name": "Walking", "path": "Walking/img", "startFrame": 1, "endFrame": 412, "nz": 4, "ext": "jpg",
         "anno_path": "Walking/groundtruth_rect.txt",
         "object_class": "person"},
        {"name": "Walking2", "path": "Walking2/img", "startFrame": 1, "endFrame": 500, "nz": 4, "ext": "jpg",
         "anno_path": "Walking2/groundtruth_rect.txt",
         "object_class": "person"},
        {"name": "Woman", "path": "Woman/img", "startFrame": 1, "endFrame": 597, "nz": 4, "ext": "jpg",
         "anno_path": "Woman/groundtruth_rect.txt",
         "object_class": "person"}
    ]

    name_list = sorted([seq['name'] for seq in sequence_info_list])
    video_name_list = [n for n in video_name_list if n in name_list]
    assert len(video_name_list) == 100

    video_list = []
    for sequence_info in tqdm(sequence_info_list, ascii=True, desc='loading OTB'):
        v_n = sequence_info['name']
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        ims = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(
            base_path=video_root, sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext)
            for frame_num in range(start_frame + init_omit, end_frame + 1)]

        gt_path = '{}/{}'.format(annos_root, sequence_info['anno_path'])

        # NOTE: OTB has some weird annos which panda cannot handle
        try:
            gts = np.loadtxt(gt_path, delimiter=',').reshape(-1, 4).tolist()  # [x y w h]
        except:
            gts = np.loadtxt(gt_path, delimiter='\t').reshape(-1, 4).tolist()  # [x y w h]

        video_list.append([v_n, ims, gts, ''])

    return video_list


def load_nfs(root):
    video_root = root#os.path.join(root, 'sequences')
    annos_root = root

    video_name_list = os.listdir(video_root)
    video_name_list = [v for v in video_name_list if os.path.isdir(os.path.join(video_root, v))]
    video_name_list = sorted(video_name_list)

    sequence_info_list = [
        {"name": "nfs_Gymnastics", "path": "sequences/Gymnastics", "startFrame": 1, "endFrame": 368, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_Gymnastics.txt", "object_class": "person", 'occlusion': False},
        {"name": "nfs_MachLoop_jet", "path": "sequences/MachLoop_jet", "startFrame": 1, "endFrame": 99, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_MachLoop_jet.txt", "object_class": "aircraft", 'occlusion': False},
        {"name": "nfs_Skiing_red", "path": "sequences/Skiing_red", "startFrame": 1, "endFrame": 69, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_Skiing_red.txt", "object_class": "person", 'occlusion': False},
        {"name": "nfs_Skydiving", "path": "sequences/Skydiving", "startFrame": 1, "endFrame": 196, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_Skydiving.txt", "object_class": "person", 'occlusion': True},
        {"name": "nfs_airboard_1", "path": "sequences/airboard_1", "startFrame": 1, "endFrame": 425, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_airboard_1.txt", "object_class": "ball", 'occlusion': False},
        {"name": "nfs_airplane_landing", "path": "sequences/airplane_landing", "startFrame": 1, "endFrame": 81, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_airplane_landing.txt", "object_class": "aircraft", 'occlusion': False},
        {"name": "nfs_airtable_3", "path": "sequences/airtable_3", "startFrame": 1, "endFrame": 482, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_airtable_3.txt", "object_class": "ball", 'occlusion': False},
        {"name": "nfs_basketball_1", "path": "sequences/basketball_1", "startFrame": 1, "endFrame": 282, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_basketball_1.txt", "object_class": "ball", 'occlusion': False},
        {"name": "nfs_basketball_2", "path": "sequences/basketball_2", "startFrame": 1, "endFrame": 102, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_basketball_2.txt", "object_class": "ball", 'occlusion': False},
        {"name": "nfs_basketball_3", "path": "sequences/basketball_3", "startFrame": 1, "endFrame": 421, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_basketball_3.txt", "object_class": "ball", 'occlusion': False},
        {"name": "nfs_basketball_6", "path": "sequences/basketball_6", "startFrame": 1, "endFrame": 224, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_basketball_6.txt", "object_class": "ball", 'occlusion': False},
        {"name": "nfs_basketball_7", "path": "sequences/basketball_7", "startFrame": 1, "endFrame": 240, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_basketball_7.txt", "object_class": "person", 'occlusion': True},
        {"name": "nfs_basketball_player", "path": "sequences/basketball_player", "startFrame": 1, "endFrame": 369,
         "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_basketball_player.txt", "object_class": "person",
         'occlusion': True},
        {"name": "nfs_basketball_player_2", "path": "sequences/basketball_player_2", "startFrame": 1, "endFrame": 437,
         "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_basketball_player_2.txt", "object_class": "person",
         'occlusion': False},
        {"name": "nfs_beach_flipback_person", "path": "sequences/beach_flipback_person", "startFrame": 1,
         "endFrame": 61, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_beach_flipback_person.txt",
         "object_class": "person head", 'occlusion': False},
        {"name": "nfs_bee", "path": "sequences/bee", "startFrame": 1, "endFrame": 45, "nz": 5, "ext": "jpg",
         "anno_path": "anno/nfs_bee.txt", "object_class": "insect", 'occlusion': False},
        {"name": "nfs_biker_acrobat", "path": "sequences/biker_acrobat", "startFrame": 1, "endFrame": 128, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_biker_acrobat.txt", "object_class": "bicycle", 'occlusion': False},
        {"name": "nfs_biker_all_1", "path": "sequences/biker_all_1", "startFrame": 1, "endFrame": 113, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_biker_all_1.txt", "object_class": "person", 'occlusion': False},
        {"name": "nfs_biker_head_2", "path": "sequences/biker_head_2", "startFrame": 1, "endFrame": 132, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_biker_head_2.txt", "object_class": "person head", 'occlusion': False},
        {"name": "nfs_biker_head_3", "path": "sequences/biker_head_3", "startFrame": 1, "endFrame": 254, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_biker_head_3.txt", "object_class": "person head", 'occlusion': False},
        {"name": "nfs_biker_upper_body", "path": "sequences/biker_upper_body", "startFrame": 1, "endFrame": 194,
         "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_biker_upper_body.txt", "object_class": "person",
         'occlusion': False},
        {"name": "nfs_biker_whole_body", "path": "sequences/biker_whole_body", "startFrame": 1, "endFrame": 572,
         "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_biker_whole_body.txt", "object_class": "person",
         'occlusion': True},
        {"name": "nfs_billiard_2", "path": "sequences/billiard_2", "startFrame": 1, "endFrame": 604, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_billiard_2.txt", "object_class": "ball", 'occlusion': False},
        {"name": "nfs_billiard_3", "path": "sequences/billiard_3", "startFrame": 1, "endFrame": 698, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_billiard_3.txt", "object_class": "ball", 'occlusion': False},
        {"name": "nfs_billiard_6", "path": "sequences/billiard_6", "startFrame": 1, "endFrame": 771, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_billiard_6.txt", "object_class": "ball", 'occlusion': False},
        {"name": "nfs_billiard_7", "path": "sequences/billiard_7", "startFrame": 1, "endFrame": 724, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_billiard_7.txt", "object_class": "ball", 'occlusion': False},
        {"name": "nfs_billiard_8", "path": "sequences/billiard_8", "startFrame": 1, "endFrame": 778, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_billiard_8.txt", "object_class": "ball", 'occlusion': False},
        {"name": "nfs_bird_2", "path": "sequences/bird_2", "startFrame": 1, "endFrame": 476, "nz": 5, "ext": "jpg",
         "anno_path": "anno/nfs_bird_2.txt", "object_class": "bird", 'occlusion': False},
        {"name": "nfs_book", "path": "sequences/book", "startFrame": 1, "endFrame": 288, "nz": 5, "ext": "jpg",
         "anno_path": "anno/nfs_book.txt", "object_class": "other", 'occlusion': False},
        {"name": "nfs_bottle", "path": "sequences/bottle", "startFrame": 1, "endFrame": 2103, "nz": 5, "ext": "jpg",
         "anno_path": "anno/nfs_bottle.txt", "object_class": "other", 'occlusion': False},
        {"name": "nfs_bowling_1", "path": "sequences/bowling_1", "startFrame": 1, "endFrame": 303, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_bowling_1.txt", "object_class": "ball", 'occlusion': True},
        {"name": "nfs_bowling_2", "path": "sequences/bowling_2", "startFrame": 1, "endFrame": 710, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_bowling_2.txt", "object_class": "ball", 'occlusion': True},
        {"name": "nfs_bowling_3", "path": "sequences/bowling_3", "startFrame": 1, "endFrame": 271, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_bowling_3.txt", "object_class": "ball", 'occlusion': True},
        {"name": "nfs_bowling_6", "path": "sequences/bowling_6", "startFrame": 1, "endFrame": 260, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_bowling_6.txt", "object_class": "ball", 'occlusion': False},
        {"name": "nfs_bowling_ball", "path": "sequences/bowling_ball", "startFrame": 1, "endFrame": 275, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_bowling_ball.txt", "object_class": "ball", 'occlusion': True},
        {"name": "nfs_bunny", "path": "sequences/bunny", "startFrame": 1, "endFrame": 705, "nz": 5, "ext": "jpg",
         "anno_path": "anno/nfs_bunny.txt", "object_class": "mammal", 'occlusion': False},
        {"name": "nfs_car", "path": "sequences/car", "startFrame": 1, "endFrame": 2020, "nz": 5, "ext": "jpg",
         "anno_path": "anno/nfs_car.txt", "object_class": "car", 'occlusion': True},
        {"name": "nfs_car_camaro", "path": "sequences/car_camaro", "startFrame": 1, "endFrame": 36, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_car_camaro.txt", "object_class": "car", 'occlusion': False},
        {"name": "nfs_car_drifting", "path": "sequences/car_drifting", "startFrame": 1, "endFrame": 173, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_car_drifting.txt", "object_class": "car", 'occlusion': False},
        {"name": "nfs_car_jumping", "path": "sequences/car_jumping", "startFrame": 1, "endFrame": 22, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_car_jumping.txt", "object_class": "car", 'occlusion': False},
        {"name": "nfs_car_rc_rolling", "path": "sequences/car_rc_rolling", "startFrame": 1, "endFrame": 62, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_car_rc_rolling.txt", "object_class": "car", 'occlusion': False},
        {"name": "nfs_car_rc_rotating", "path": "sequences/car_rc_rotating", "startFrame": 1, "endFrame": 80, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_car_rc_rotating.txt", "object_class": "car", 'occlusion': False},
        {"name": "nfs_car_side", "path": "sequences/car_side", "startFrame": 1, "endFrame": 108, "nz": 5, "ext": "jpg",
         "anno_path": "anno/nfs_car_side.txt", "object_class": "car", 'occlusion': False},
        {"name": "nfs_car_white", "path": "sequences/car_white", "startFrame": 1, "endFrame": 2063, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_car_white.txt", "object_class": "car", 'occlusion': False},
        {"name": "nfs_cheetah", "path": "sequences/cheetah", "startFrame": 1, "endFrame": 167, "nz": 5, "ext": "jpg",
         "anno_path": "anno/nfs_cheetah.txt", "object_class": "mammal", 'occlusion': True},
        {"name": "nfs_cup", "path": "sequences/cup", "startFrame": 1, "endFrame": 1281, "nz": 5, "ext": "jpg",
         "anno_path": "anno/nfs_cup.txt", "object_class": "other", 'occlusion': False},
        {"name": "nfs_cup_2", "path": "sequences/cup_2", "startFrame": 1, "endFrame": 182, "nz": 5, "ext": "jpg",
         "anno_path": "anno/nfs_cup_2.txt", "object_class": "other", 'occlusion': False},
        {"name": "nfs_dog", "path": "sequences/dog", "startFrame": 1, "endFrame": 1030, "nz": 5, "ext": "jpg",
         "anno_path": "anno/nfs_dog.txt", "object_class": "dog", 'occlusion': True},
        {"name": "nfs_dog_1", "path": "sequences/dog_1", "startFrame": 1, "endFrame": 168, "nz": 5, "ext": "jpg",
         "anno_path": "anno/nfs_dog_1.txt", "object_class": "dog", 'occlusion': False},
        {"name": "nfs_dog_2", "path": "sequences/dog_2", "startFrame": 1, "endFrame": 594, "nz": 5, "ext": "jpg",
         "anno_path": "anno/nfs_dog_2.txt", "object_class": "dog", 'occlusion': True},
        {"name": "nfs_dog_3", "path": "sequences/dog_3", "startFrame": 1, "endFrame": 200, "nz": 5, "ext": "jpg",
         "anno_path": "anno/nfs_dog_3.txt", "object_class": "dog", 'occlusion': False},
        {"name": "nfs_dogs", "path": "sequences/dogs", "startFrame": 1, "endFrame": 198, "nz": 5, "ext": "jpg",
         "anno_path": "anno/nfs_dogs.txt", "object_class": "dog", 'occlusion': True},
        {"name": "nfs_dollar", "path": "sequences/dollar", "startFrame": 1, "endFrame": 1426, "nz": 5, "ext": "jpg",
         "anno_path": "anno/nfs_dollar.txt", "object_class": "other", 'occlusion': False},
        {"name": "nfs_drone", "path": "sequences/drone", "startFrame": 1, "endFrame": 70, "nz": 5, "ext": "jpg",
         "anno_path": "anno/nfs_drone.txt", "object_class": "aircraft", 'occlusion': False},
        {"name": "nfs_ducks_lake", "path": "sequences/ducks_lake", "startFrame": 1, "endFrame": 107, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_ducks_lake.txt", "object_class": "bird", 'occlusion': False},
        {"name": "nfs_exit", "path": "sequences/exit", "startFrame": 1, "endFrame": 359, "nz": 5, "ext": "jpg",
         "anno_path": "anno/nfs_exit.txt", "object_class": "other", 'occlusion': False},
        {"name": "nfs_first", "path": "sequences/first", "startFrame": 1, "endFrame": 435, "nz": 5, "ext": "jpg",
         "anno_path": "anno/nfs_first.txt", "object_class": "other", 'occlusion': False},
        {"name": "nfs_flower", "path": "sequences/flower", "startFrame": 1, "endFrame": 448, "nz": 5, "ext": "jpg",
         "anno_path": "anno/nfs_flower.txt", "object_class": "other", 'occlusion': False},
        {"name": "nfs_footbal_skill", "path": "sequences/footbal_skill", "startFrame": 1, "endFrame": 131, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_footbal_skill.txt", "object_class": "ball", 'occlusion': True},
        {"name": "nfs_helicopter", "path": "sequences/helicopter", "startFrame": 1, "endFrame": 310, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_helicopter.txt", "object_class": "aircraft", 'occlusion': False},
        {"name": "nfs_horse_jumping", "path": "sequences/horse_jumping", "startFrame": 1, "endFrame": 117, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_horse_jumping.txt", "object_class": "horse", 'occlusion': True},
        {"name": "nfs_horse_running", "path": "sequences/horse_running", "startFrame": 1, "endFrame": 139, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_horse_running.txt", "object_class": "horse", 'occlusion': False},
        {"name": "nfs_iceskating_6", "path": "sequences/iceskating_6", "startFrame": 1, "endFrame": 603, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_iceskating_6.txt", "object_class": "person", 'occlusion': False},
        {"name": "nfs_jellyfish_5", "path": "sequences/jellyfish_5", "startFrame": 1, "endFrame": 746, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_jellyfish_5.txt", "object_class": "invertebrate", 'occlusion': False},
        {"name": "nfs_kid_swing", "path": "sequences/kid_swing", "startFrame": 1, "endFrame": 169, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_kid_swing.txt", "object_class": "person", 'occlusion': False},
        {"name": "nfs_motorcross", "path": "sequences/motorcross", "startFrame": 1, "endFrame": 39, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_motorcross.txt", "object_class": "vehicle", 'occlusion': True},
        {"name": "nfs_motorcross_kawasaki", "path": "sequences/motorcross_kawasaki", "startFrame": 1, "endFrame": 65,
         "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_motorcross_kawasaki.txt", "object_class": "vehicle",
         'occlusion': False},
        {"name": "nfs_parkour", "path": "sequences/parkour", "startFrame": 1, "endFrame": 58, "nz": 5, "ext": "jpg",
         "anno_path": "anno/nfs_parkour.txt", "object_class": "person head", 'occlusion': False},
        {"name": "nfs_person_scooter", "path": "sequences/person_scooter", "startFrame": 1, "endFrame": 413, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_person_scooter.txt", "object_class": "person", 'occlusion': True},
        {"name": "nfs_pingpong_2", "path": "sequences/pingpong_2", "startFrame": 1, "endFrame": 1277, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_pingpong_2.txt", "object_class": "ball", 'occlusion': False},
        {"name": "nfs_pingpong_7", "path": "sequences/pingpong_7", "startFrame": 1, "endFrame": 1290, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_pingpong_7.txt", "object_class": "ball", 'occlusion': False},
        {"name": "nfs_pingpong_8", "path": "sequences/pingpong_8", "startFrame": 1, "endFrame": 296, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_pingpong_8.txt", "object_class": "ball", 'occlusion': False},
        {"name": "nfs_purse", "path": "sequences/purse", "startFrame": 1, "endFrame": 968, "nz": 5, "ext": "jpg",
         "anno_path": "anno/nfs_purse.txt", "object_class": "other", 'occlusion': False},
        {"name": "nfs_rubber", "path": "sequences/rubber", "startFrame": 1, "endFrame": 1328, "nz": 5, "ext": "jpg",
         "anno_path": "anno/nfs_rubber.txt", "object_class": "other", 'occlusion': False},
        {"name": "nfs_running", "path": "sequences/running", "startFrame": 1, "endFrame": 677, "nz": 5, "ext": "jpg",
         "anno_path": "anno/nfs_running.txt", "object_class": "person", 'occlusion': False},
        {"name": "nfs_running_100_m", "path": "sequences/running_100_m", "startFrame": 1, "endFrame": 313, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_running_100_m.txt", "object_class": "person", 'occlusion': True},
        {"name": "nfs_running_100_m_2", "path": "sequences/running_100_m_2", "startFrame": 1, "endFrame": 337, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_running_100_m_2.txt", "object_class": "person", 'occlusion': True},
        {"name": "nfs_running_2", "path": "sequences/running_2", "startFrame": 1, "endFrame": 363, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_running_2.txt", "object_class": "person", 'occlusion': False},
        {"name": "nfs_shuffleboard_1", "path": "sequences/shuffleboard_1", "startFrame": 1, "endFrame": 42, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_shuffleboard_1.txt", "object_class": "other", 'occlusion': False},
        {"name": "nfs_shuffleboard_2", "path": "sequences/shuffleboard_2", "startFrame": 1, "endFrame": 41, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_shuffleboard_2.txt", "object_class": "other", 'occlusion': False},
        {"name": "nfs_shuffleboard_4", "path": "sequences/shuffleboard_4", "startFrame": 1, "endFrame": 62, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_shuffleboard_4.txt", "object_class": "other", 'occlusion': False},
        {"name": "nfs_shuffleboard_5", "path": "sequences/shuffleboard_5", "startFrame": 1, "endFrame": 32, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_shuffleboard_5.txt", "object_class": "other", 'occlusion': False},
        {"name": "nfs_shuffleboard_6", "path": "sequences/shuffleboard_6", "startFrame": 1, "endFrame": 52, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_shuffleboard_6.txt", "object_class": "other", 'occlusion': False},
        {"name": "nfs_shuffletable_2", "path": "sequences/shuffletable_2", "startFrame": 1, "endFrame": 372, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_shuffletable_2.txt", "object_class": "other", 'occlusion': False},
        {"name": "nfs_shuffletable_3", "path": "sequences/shuffletable_3", "startFrame": 1, "endFrame": 368, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_shuffletable_3.txt", "object_class": "other", 'occlusion': False},
        {"name": "nfs_shuffletable_4", "path": "sequences/shuffletable_4", "startFrame": 1, "endFrame": 101, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_shuffletable_4.txt", "object_class": "other", 'occlusion': False},
        {"name": "nfs_ski_long", "path": "sequences/ski_long", "startFrame": 1, "endFrame": 274, "nz": 5, "ext": "jpg",
         "anno_path": "anno/nfs_ski_long.txt", "object_class": "person", 'occlusion': False},
        {"name": "nfs_soccer_ball", "path": "sequences/soccer_ball", "startFrame": 1, "endFrame": 163, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_soccer_ball.txt", "object_class": "ball", 'occlusion': False},
        {"name": "nfs_soccer_ball_2", "path": "sequences/soccer_ball_2", "startFrame": 1, "endFrame": 1934, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_soccer_ball_2.txt", "object_class": "ball", 'occlusion': False},
        {"name": "nfs_soccer_ball_3", "path": "sequences/soccer_ball_3", "startFrame": 1, "endFrame": 1381, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_soccer_ball_3.txt", "object_class": "ball", 'occlusion': False},
        {"name": "nfs_soccer_player_2", "path": "sequences/soccer_player_2", "startFrame": 1, "endFrame": 475, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_soccer_player_2.txt", "object_class": "person", 'occlusion': False},
        {"name": "nfs_soccer_player_3", "path": "sequences/soccer_player_3", "startFrame": 1, "endFrame": 319, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_soccer_player_3.txt", "object_class": "person", 'occlusion': True},
        {"name": "nfs_stop_sign", "path": "sequences/stop_sign", "startFrame": 1, "endFrame": 302, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_stop_sign.txt", "object_class": "other", 'occlusion': False},
        {"name": "nfs_suv", "path": "sequences/suv", "startFrame": 1, "endFrame": 2584, "nz": 5, "ext": "jpg",
         "anno_path": "anno/nfs_suv.txt", "object_class": "car", 'occlusion': False},
        {"name": "nfs_tiger", "path": "sequences/tiger", "startFrame": 1, "endFrame": 1556, "nz": 5, "ext": "jpg",
         "anno_path": "anno/nfs_tiger.txt", "object_class": "mammal", 'occlusion': False},
        {"name": "nfs_walking", "path": "sequences/walking", "startFrame": 1, "endFrame": 555, "nz": 5, "ext": "jpg",
         "anno_path": "anno/nfs_walking.txt", "object_class": "person", 'occlusion': False},
        {"name": "nfs_walking_3", "path": "sequences/walking_3", "startFrame": 1, "endFrame": 1427, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_walking_3.txt", "object_class": "person", 'occlusion': False},
        {"name": "nfs_water_ski_2", "path": "sequences/water_ski_2", "startFrame": 1, "endFrame": 47, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_water_ski_2.txt", "object_class": "person", 'occlusion': False},
        {"name": "nfs_yoyo", "path": "sequences/yoyo", "startFrame": 1, "endFrame": 67, "nz": 5, "ext": "jpg",
         "anno_path": "anno/nfs_yoyo.txt", "object_class": "other", 'occlusion': False},
        {"name": "nfs_zebra_fish", "path": "sequences/zebra_fish", "startFrame": 1, "endFrame": 671, "nz": 5,
         "ext": "jpg", "anno_path": "anno/nfs_zebra_fish.txt", "object_class": "fish", 'occlusion': False},
    ]

    video_name_list = sorted([seq['name'] for seq in sequence_info_list])
    # video_name_list = [n for n in video_name_list if n in name_list]
    assert len(video_name_list) == 100

    video_list = []
    for sequence_info in tqdm(sequence_info_list, ascii=True, desc='loading NFS'):
        v_n = sequence_info['name']
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        ims = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(
            base_path=video_root, sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext)
            for frame_num in range(start_frame + init_omit, end_frame + 1)]

        gt_path = '{}/{}'.format(annos_root, sequence_info['anno_path'])

        # NOTE: OTB has some weird annos which panda cannot handle
        try:
            gts = np.loadtxt(gt_path, delimiter=',').reshape(-1, 4).tolist()  # [x y w h]
        except:
            gts = np.loadtxt(gt_path, delimiter='\t').reshape(-1, 4).tolist()  # [x y w h]

        video_list.append([v_n, ims, gts, ''])

    return video_list


def load_uav(root):
    video_root = root#os.path.join(root, 'sequences')
    annos_root = root

    video_name_list = os.listdir(video_root)
    video_name_list = [v for v in video_name_list if os.path.isdir(os.path.join(video_root, v))]
    video_name_list = sorted(video_name_list)

    sequence_info_list = [
        {"name": "uav_bike1", "path": "data_seq/UAV123/bike1", "startFrame": 1, "endFrame": 3085, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/bike1.txt", "object_class": "vehicle"},
        {"name": "uav_bike2", "path": "data_seq/UAV123/bike2", "startFrame": 1, "endFrame": 553, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/bike2.txt", "object_class": "vehicle"},
        {"name": "uav_bike3", "path": "data_seq/UAV123/bike3", "startFrame": 1, "endFrame": 433, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/bike3.txt", "object_class": "vehicle"},
        {"name": "uav_bird1_1", "path": "data_seq/UAV123/bird1", "startFrame": 1, "endFrame": 253, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/bird1_1.txt", "object_class": "bird"},
        {"name": "uav_bird1_2", "path": "data_seq/UAV123/bird1", "startFrame": 775, "endFrame": 1477, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/bird1_2.txt", "object_class": "bird"},
        {"name": "uav_bird1_3", "path": "data_seq/UAV123/bird1", "startFrame": 1573, "endFrame": 2437, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/bird1_3.txt", "object_class": "bird"},
        {"name": "uav_boat1", "path": "data_seq/UAV123/boat1", "startFrame": 1, "endFrame": 901, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/boat1.txt", "object_class": "vessel"},
        {"name": "uav_boat2", "path": "data_seq/UAV123/boat2", "startFrame": 1, "endFrame": 799, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/boat2.txt", "object_class": "vessel"},
        {"name": "uav_boat3", "path": "data_seq/UAV123/boat3", "startFrame": 1, "endFrame": 901, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/boat3.txt", "object_class": "vessel"},
        {"name": "uav_boat4", "path": "data_seq/UAV123/boat4", "startFrame": 1, "endFrame": 553, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/boat4.txt", "object_class": "vessel"},
        {"name": "uav_boat5", "path": "data_seq/UAV123/boat5", "startFrame": 1, "endFrame": 505, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/boat5.txt", "object_class": "vessel"},
        {"name": "uav_boat6", "path": "data_seq/UAV123/boat6", "startFrame": 1, "endFrame": 805, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/boat6.txt", "object_class": "vessel"},
        {"name": "uav_boat7", "path": "data_seq/UAV123/boat7", "startFrame": 1, "endFrame": 535, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/boat7.txt", "object_class": "vessel"},
        {"name": "uav_boat8", "path": "data_seq/UAV123/boat8", "startFrame": 1, "endFrame": 685, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/boat8.txt", "object_class": "vessel"},
        {"name": "uav_boat9", "path": "data_seq/UAV123/boat9", "startFrame": 1, "endFrame": 1399, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/boat9.txt", "object_class": "vessel"},
        {"name": "uav_building1", "path": "data_seq/UAV123/building1", "startFrame": 1, "endFrame": 469, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/building1.txt", "object_class": "other"},
        {"name": "uav_building2", "path": "data_seq/UAV123/building2", "startFrame": 1, "endFrame": 577, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/building2.txt", "object_class": "other"},
        {"name": "uav_building3", "path": "data_seq/UAV123/building3", "startFrame": 1, "endFrame": 829, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/building3.txt", "object_class": "other"},
        {"name": "uav_building4", "path": "data_seq/UAV123/building4", "startFrame": 1, "endFrame": 787, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/building4.txt", "object_class": "other"},
        {"name": "uav_building5", "path": "data_seq/UAV123/building5", "startFrame": 1, "endFrame": 481, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/building5.txt", "object_class": "other"},
        {"name": "uav_car1_1", "path": "data_seq/UAV123/car1", "startFrame": 1, "endFrame": 751, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/car1_1.txt", "object_class": "car"},
        {"name": "uav_car1_2", "path": "data_seq/UAV123/car1", "startFrame": 751, "endFrame": 1627, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/car1_2.txt", "object_class": "car"},
        {"name": "uav_car1_3", "path": "data_seq/UAV123/car1", "startFrame": 1627, "endFrame": 2629, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/car1_3.txt", "object_class": "car"},
        {"name": "uav_car10", "path": "data_seq/UAV123/car10", "startFrame": 1, "endFrame": 1405, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/car10.txt", "object_class": "car"},
        {"name": "uav_car11", "path": "data_seq/UAV123/car11", "startFrame": 1, "endFrame": 337, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/car11.txt", "object_class": "car"},
        {"name": "uav_car12", "path": "data_seq/UAV123/car12", "startFrame": 1, "endFrame": 499, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/car12.txt", "object_class": "car"},
        {"name": "uav_car13", "path": "data_seq/UAV123/car13", "startFrame": 1, "endFrame": 415, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/car13.txt", "object_class": "car"},
        {"name": "uav_car14", "path": "data_seq/UAV123/car14", "startFrame": 1, "endFrame": 1327, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/car14.txt", "object_class": "car"},
        {"name": "uav_car15", "path": "data_seq/UAV123/car15", "startFrame": 1, "endFrame": 469, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/car15.txt", "object_class": "car"},
        {"name": "uav_car16_1", "path": "data_seq/UAV123/car16", "startFrame": 1, "endFrame": 415, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/car16_1.txt", "object_class": "car"},
        {"name": "uav_car16_2", "path": "data_seq/UAV123/car16", "startFrame": 415, "endFrame": 1993, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/car16_2.txt", "object_class": "car"},
        {"name": "uav_car17", "path": "data_seq/UAV123/car17", "startFrame": 1, "endFrame": 1057, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/car17.txt", "object_class": "car"},
        {"name": "uav_car18", "path": "data_seq/UAV123/car18", "startFrame": 1, "endFrame": 1207, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/car18.txt", "object_class": "car"},
        {"name": "uav_car1_s", "path": "data_seq/UAV123/car1_s", "startFrame": 1, "endFrame": 1475, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/car1_s.txt", "object_class": "car"},
        {"name": "uav_car2", "path": "data_seq/UAV123/car2", "startFrame": 1, "endFrame": 1321, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/car2.txt", "object_class": "car"},
        {"name": "uav_car2_s", "path": "data_seq/UAV123/car2_s", "startFrame": 1, "endFrame": 320, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/car2_s.txt", "object_class": "car"},
        {"name": "uav_car3", "path": "data_seq/UAV123/car3", "startFrame": 1, "endFrame": 1717, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/car3.txt", "object_class": "car"},
        {"name": "uav_car3_s", "path": "data_seq/UAV123/car3_s", "startFrame": 1, "endFrame": 1300, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/car3_s.txt", "object_class": "car"},
        {"name": "uav_car4", "path": "data_seq/UAV123/car4", "startFrame": 1, "endFrame": 1345, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/car4.txt", "object_class": "car"},
        {"name": "uav_car4_s", "path": "data_seq/UAV123/car4_s", "startFrame": 1, "endFrame": 830, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/car4_s.txt", "object_class": "car"},
        {"name": "uav_car5", "path": "data_seq/UAV123/car5", "startFrame": 1, "endFrame": 745, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/car5.txt", "object_class": "car"},
        {"name": "uav_car6_1", "path": "data_seq/UAV123/car6", "startFrame": 1, "endFrame": 487, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/car6_1.txt", "object_class": "car"},
        {"name": "uav_car6_2", "path": "data_seq/UAV123/car6", "startFrame": 487, "endFrame": 1807, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/car6_2.txt", "object_class": "car"},
        {"name": "uav_car6_3", "path": "data_seq/UAV123/car6", "startFrame": 1807, "endFrame": 2953, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/car6_3.txt", "object_class": "car"},
        {"name": "uav_car6_4", "path": "data_seq/UAV123/car6", "startFrame": 2953, "endFrame": 3925, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/car6_4.txt", "object_class": "car"},
        {"name": "uav_car6_5", "path": "data_seq/UAV123/car6", "startFrame": 3925, "endFrame": 4861, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/car6_5.txt", "object_class": "car"},
        {"name": "uav_car7", "path": "data_seq/UAV123/car7", "startFrame": 1, "endFrame": 1033, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/car7.txt", "object_class": "car"},
        {"name": "uav_car8_1", "path": "data_seq/UAV123/car8", "startFrame": 1, "endFrame": 1357, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/car8_1.txt", "object_class": "car"},
        {"name": "uav_car8_2", "path": "data_seq/UAV123/car8", "startFrame": 1357, "endFrame": 2575, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/car8_2.txt", "object_class": "car"},
        {"name": "uav_car9", "path": "data_seq/UAV123/car9", "startFrame": 1, "endFrame": 1879, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/car9.txt", "object_class": "car"},
        {"name": "uav_group1_1", "path": "data_seq/UAV123/group1", "startFrame": 1, "endFrame": 1333, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/group1_1.txt", "object_class": "person"},
        {"name": "uav_group1_2", "path": "data_seq/UAV123/group1", "startFrame": 1333, "endFrame": 2515, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/group1_2.txt", "object_class": "person"},
        {"name": "uav_group1_3", "path": "data_seq/UAV123/group1", "startFrame": 2515, "endFrame": 3925, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/group1_3.txt", "object_class": "person"},
        {"name": "uav_group1_4", "path": "data_seq/UAV123/group1", "startFrame": 3925, "endFrame": 4873, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/group1_4.txt", "object_class": "person"},
        {"name": "uav_group2_1", "path": "data_seq/UAV123/group2", "startFrame": 1, "endFrame": 907, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/group2_1.txt", "object_class": "person"},
        {"name": "uav_group2_2", "path": "data_seq/UAV123/group2", "startFrame": 907, "endFrame": 1771, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/group2_2.txt", "object_class": "person"},
        {"name": "uav_group2_3", "path": "data_seq/UAV123/group2", "startFrame": 1771, "endFrame": 2683, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/group2_3.txt", "object_class": "person"},
        {"name": "uav_group3_1", "path": "data_seq/UAV123/group3", "startFrame": 1, "endFrame": 1567, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/group3_1.txt", "object_class": "person"},
        {"name": "uav_group3_2", "path": "data_seq/UAV123/group3", "startFrame": 1567, "endFrame": 2827, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/group3_2.txt", "object_class": "person"},
        {"name": "uav_group3_3", "path": "data_seq/UAV123/group3", "startFrame": 2827, "endFrame": 4369, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/group3_3.txt", "object_class": "person"},
        {"name": "uav_group3_4", "path": "data_seq/UAV123/group3", "startFrame": 4369, "endFrame": 5527, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/group3_4.txt", "object_class": "person"},
        {"name": "uav_person1", "path": "data_seq/UAV123/person1", "startFrame": 1, "endFrame": 799, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/person1.txt", "object_class": "person"},
        {"name": "uav_person10", "path": "data_seq/UAV123/person10", "startFrame": 1, "endFrame": 1021, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/person10.txt", "object_class": "person"},
        {"name": "uav_person11", "path": "data_seq/UAV123/person11", "startFrame": 1, "endFrame": 721, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/person11.txt", "object_class": "person"},
        {"name": "uav_person12_1", "path": "data_seq/UAV123/person12", "startFrame": 1, "endFrame": 601, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/person12_1.txt", "object_class": "person"},
        {"name": "uav_person12_2", "path": "data_seq/UAV123/person12", "startFrame": 601, "endFrame": 1621, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/person12_2.txt", "object_class": "person"},
        {"name": "uav_person13", "path": "data_seq/UAV123/person13", "startFrame": 1, "endFrame": 883, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/person13.txt", "object_class": "person"},
        {"name": "uav_person14_1", "path": "data_seq/UAV123/person14", "startFrame": 1, "endFrame": 847, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/person14_1.txt", "object_class": "person"},
        {"name": "uav_person14_2", "path": "data_seq/UAV123/person14", "startFrame": 847, "endFrame": 1813, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/person14_2.txt", "object_class": "person"},
        {"name": "uav_person14_3", "path": "data_seq/UAV123/person14", "startFrame": 1813, "endFrame": 2923,
         "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person14_3.txt", "object_class": "person"},
        {"name": "uav_person15", "path": "data_seq/UAV123/person15", "startFrame": 1, "endFrame": 1339, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/person15.txt", "object_class": "person"},
        {"name": "uav_person16", "path": "data_seq/UAV123/person16", "startFrame": 1, "endFrame": 1147, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/person16.txt", "object_class": "person"},
        {"name": "uav_person17_1", "path": "data_seq/UAV123/person17", "startFrame": 1, "endFrame": 1501, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/person17_1.txt", "object_class": "person"},
        {"name": "uav_person17_2", "path": "data_seq/UAV123/person17", "startFrame": 1501, "endFrame": 2347,
         "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person17_2.txt", "object_class": "person"},
        {"name": "uav_person18", "path": "data_seq/UAV123/person18", "startFrame": 1, "endFrame": 1393, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/person18.txt", "object_class": "person"},
        {"name": "uav_person19_1", "path": "data_seq/UAV123/person19", "startFrame": 1, "endFrame": 1243, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/person19_1.txt", "object_class": "person"},
        {"name": "uav_person19_2", "path": "data_seq/UAV123/person19", "startFrame": 1243, "endFrame": 2791,
         "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person19_2.txt", "object_class": "person"},
        {"name": "uav_person19_3", "path": "data_seq/UAV123/person19", "startFrame": 2791, "endFrame": 4357,
         "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person19_3.txt", "object_class": "person"},
        {"name": "uav_person1_s", "path": "data_seq/UAV123/person1_s", "startFrame": 1, "endFrame": 1600, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/person1_s.txt", "object_class": "person"},
        {"name": "uav_person2_1", "path": "data_seq/UAV123/person2", "startFrame": 1, "endFrame": 1189, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/person2_1.txt", "object_class": "person"},
        {"name": "uav_person2_2", "path": "data_seq/UAV123/person2", "startFrame": 1189, "endFrame": 2623, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/person2_2.txt", "object_class": "person"},
        {"name": "uav_person20", "path": "data_seq/UAV123/person20", "startFrame": 1, "endFrame": 1783, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/person20.txt", "object_class": "person"},
        {"name": "uav_person21", "path": "data_seq/UAV123/person21", "startFrame": 1, "endFrame": 487, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/person21.txt", "object_class": "person"},
        {"name": "uav_person22", "path": "data_seq/UAV123/person22", "startFrame": 1, "endFrame": 199, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/person22.txt", "object_class": "person"},
        {"name": "uav_person23", "path": "data_seq/UAV123/person23", "startFrame": 1, "endFrame": 397, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/person23.txt", "object_class": "person"},
        {"name": "uav_person2_s", "path": "data_seq/UAV123/person2_s", "startFrame": 1, "endFrame": 250, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/person2_s.txt", "object_class": "person"},
        {"name": "uav_person3", "path": "data_seq/UAV123/person3", "startFrame": 1, "endFrame": 643, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/person3.txt", "object_class": "person"},
        {"name": "uav_person3_s", "path": "data_seq/UAV123/person3_s", "startFrame": 1, "endFrame": 505, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/person3_s.txt", "object_class": "person"},
        {"name": "uav_person4_1", "path": "data_seq/UAV123/person4", "startFrame": 1, "endFrame": 1501, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/person4_1.txt", "object_class": "person"},
        {"name": "uav_person4_2", "path": "data_seq/UAV123/person4", "startFrame": 1501, "endFrame": 2743, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/person4_2.txt", "object_class": "person"},
        {"name": "uav_person5_1", "path": "data_seq/UAV123/person5", "startFrame": 1, "endFrame": 877, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/person5_1.txt", "object_class": "person"},
        {"name": "uav_person5_2", "path": "data_seq/UAV123/person5", "startFrame": 877, "endFrame": 2101, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/person5_2.txt", "object_class": "person"},
        {"name": "uav_person6", "path": "data_seq/UAV123/person6", "startFrame": 1, "endFrame": 901, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/person6.txt", "object_class": "person"},
        {"name": "uav_person7_1", "path": "data_seq/UAV123/person7", "startFrame": 1, "endFrame": 1249, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/person7_1.txt", "object_class": "person"},
        {"name": "uav_person7_2", "path": "data_seq/UAV123/person7", "startFrame": 1249, "endFrame": 2065, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/person7_2.txt", "object_class": "person"},
        {"name": "uav_person8_1", "path": "data_seq/UAV123/person8", "startFrame": 1, "endFrame": 1075, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/person8_1.txt", "object_class": "person"},
        {"name": "uav_person8_2", "path": "data_seq/UAV123/person8", "startFrame": 1075, "endFrame": 1525, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/person8_2.txt", "object_class": "person"},
        {"name": "uav_person9", "path": "data_seq/UAV123/person9", "startFrame": 1, "endFrame": 661, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/person9.txt", "object_class": "person"},
        {"name": "uav_truck1", "path": "data_seq/UAV123/truck1", "startFrame": 1, "endFrame": 463, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/truck1.txt", "object_class": "truck"},
        {"name": "uav_truck2", "path": "data_seq/UAV123/truck2", "startFrame": 1, "endFrame": 385, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/truck2.txt", "object_class": "truck"},
        {"name": "uav_truck3", "path": "data_seq/UAV123/truck3", "startFrame": 1, "endFrame": 535, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/truck3.txt", "object_class": "truck"},
        {"name": "uav_truck4_1", "path": "data_seq/UAV123/truck4", "startFrame": 1, "endFrame": 577, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/truck4_1.txt", "object_class": "truck"},
        {"name": "uav_truck4_2", "path": "data_seq/UAV123/truck4", "startFrame": 577, "endFrame": 1261, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/truck4_2.txt", "object_class": "truck"},
        {"name": "uav_uav1_1", "path": "data_seq/UAV123/uav1", "startFrame": 1, "endFrame": 1555, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/uav1_1.txt", "object_class": "aircraft"},
        {"name": "uav_uav1_2", "path": "data_seq/UAV123/uav1", "startFrame": 1555, "endFrame": 2377, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/uav1_2.txt", "object_class": "aircraft"},
        {"name": "uav_uav1_3", "path": "data_seq/UAV123/uav1", "startFrame": 2473, "endFrame": 3469, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/uav1_3.txt", "object_class": "aircraft"},
        {"name": "uav_uav2", "path": "data_seq/UAV123/uav2", "startFrame": 1, "endFrame": 133, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/uav2.txt", "object_class": "aircraft"},
        {"name": "uav_uav3", "path": "data_seq/UAV123/uav3", "startFrame": 1, "endFrame": 265, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/uav3.txt", "object_class": "aircraft"},
        {"name": "uav_uav4", "path": "data_seq/UAV123/uav4", "startFrame": 1, "endFrame": 157, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/uav4.txt", "object_class": "aircraft"},
        {"name": "uav_uav5", "path": "data_seq/UAV123/uav5", "startFrame": 1, "endFrame": 139, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/uav5.txt", "object_class": "aircraft"},
        {"name": "uav_uav6", "path": "data_seq/UAV123/uav6", "startFrame": 1, "endFrame": 109, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/uav6.txt", "object_class": "aircraft"},
        {"name": "uav_uav7", "path": "data_seq/UAV123/uav7", "startFrame": 1, "endFrame": 373, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/uav7.txt", "object_class": "aircraft"},
        {"name": "uav_uav8", "path": "data_seq/UAV123/uav8", "startFrame": 1, "endFrame": 301, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/uav8.txt", "object_class": "aircraft"},
        {"name": "uav_wakeboard1", "path": "data_seq/UAV123/wakeboard1", "startFrame": 1, "endFrame": 421, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/wakeboard1.txt", "object_class": "person"},
        {"name": "uav_wakeboard10", "path": "data_seq/UAV123/wakeboard10", "startFrame": 1, "endFrame": 469,
         "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/wakeboard10.txt", "object_class": "person"},
        {"name": "uav_wakeboard2", "path": "data_seq/UAV123/wakeboard2", "startFrame": 1, "endFrame": 733, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/wakeboard2.txt", "object_class": "person"},
        {"name": "uav_wakeboard3", "path": "data_seq/UAV123/wakeboard3", "startFrame": 1, "endFrame": 823, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/wakeboard3.txt", "object_class": "person"},
        {"name": "uav_wakeboard4", "path": "data_seq/UAV123/wakeboard4", "startFrame": 1, "endFrame": 697, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/wakeboard4.txt", "object_class": "person"},
        {"name": "uav_wakeboard5", "path": "data_seq/UAV123/wakeboard5", "startFrame": 1, "endFrame": 1675, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/wakeboard5.txt", "object_class": "person"},
        {"name": "uav_wakeboard6", "path": "data_seq/UAV123/wakeboard6", "startFrame": 1, "endFrame": 1165, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/wakeboard6.txt", "object_class": "person"},
        {"name": "uav_wakeboard7", "path": "data_seq/UAV123/wakeboard7", "startFrame": 1, "endFrame": 199, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/wakeboard7.txt", "object_class": "person"},
        {"name": "uav_wakeboard8", "path": "data_seq/UAV123/wakeboard8", "startFrame": 1, "endFrame": 1543, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/wakeboard8.txt", "object_class": "person"},
        {"name": "uav_wakeboard9", "path": "data_seq/UAV123/wakeboard9", "startFrame": 1, "endFrame": 355, "nz": 6,
         "ext": "jpg", "anno_path": "anno/UAV123/wakeboard9.txt", "object_class": "person"}
    ]
    video_name_list = sorted([seq['name'] for seq in sequence_info_list])
    # video_name_list = [n for n in video_name_list if n in name_list]
    assert len(video_name_list) == 123

    video_list = []
    for sequence_info in tqdm(sequence_info_list, ascii=True, desc='loading NFS'):
        v_n = sequence_info['name']
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        ims = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(
            base_path=video_root, sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext)
            for frame_num in range(start_frame + init_omit, end_frame + 1)]

        gt_path = '{}/{}'.format(annos_root, sequence_info['anno_path'])

        # NOTE: OTB has some weird annos which panda cannot handle
        try:
            gts = np.loadtxt(gt_path, delimiter=',').reshape(-1, 4).tolist()  # [x y w h]
        except:
            gts = np.loadtxt(gt_path, delimiter='\t').reshape(-1, 4).tolist()  # [x y w h]

        video_list.append([v_n, ims, gts, ''])

    return video_list



def load_trek150(root):
    video_name_list = np.loadtxt(os.path.join(root, 'sequences.txt'), dtype=np.str).tolist()
    lang_root = '/data2/Datasets/TREK-150_language'

    verb_classes = pd.read_csv(os.path.join(root, 'EPIC_verb_classes.csv'))
    verb_id = np.array(verb_classes.verb_id)
    verb_class_key = np.array(verb_classes.class_key)
    verb_verbs = np.array(verb_classes.verbs)
    verb_verbs = [vs.strip('"[').strip(']"').replace("'", "").replace(",", "").split(' ') for vs in verb_verbs]

    noun_classes = pd.read_csv(os.path.join(root, 'EPIC_noun_classes.csv'))
    noun_id = np.array(noun_classes.noun_id)
    noun_class_key = np.array(noun_classes.class_key)
    noun_nouns = np.array(noun_classes.nouns)
    noun_nouns = [ns.strip('"[').strip(']"').replace("'", "").replace(",", "").split(' ') for ns in noun_nouns]

    video_list = []
    for v_n in tqdm(video_name_list, ascii=True, desc='loading TREK-150'):
        gt_path = os.path.join(root, v_n, 'groundtruth_rect.txt')
        verb_noun_path = os.path.join(root, v_n, 'action_target.txt')
        lang_path = os.path.join(lang_root, v_n, 'language.txt')
        im_dir = os.path.join(root, v_n, 'img')

        im_list: strList = os.listdir(im_dir)
        im_list = sorted(im_list)

        gts = np.loadtxt(gt_path, delimiter=',').tolist()  # [x y w h]
        with open(verb_noun_path, 'r') as f:
            verb_noun = f.readlines()
            verb_noun = [int(vn) for vn in verb_noun]
            action_verb = verb_class_key[verb_noun[0]]
            action_noun = noun_class_key[verb_noun[1]]
            target_noun = noun_class_key[verb_noun[2]]
            lang = '{} {}, {}'.format(action_verb, action_noun, target_noun)

        with open(lang_path, 'r') as f:
            lang = f.readline()

        ims = [os.path.join(im_dir, im_f) for im_f in im_list if 'jpg' in im_f]

        video_list.append([v_n, ims, gts, lang])

    return video_list


if __name__=='__main__':
    load_uav('/data2/Documents/Datasets/UAV123')