import shutil
import os
import glob
import json
import numpy as np
from tqdm import tqdm

from toolkit.datasets.dataset import Dataset

from pysot.utils.bbox import get_axis_aligned_bbox, cxy_wh_2_rect

def attr_mapping():
    attr_maps = {"VN": {"name": "Video Number", "d_type": str}, 
                 "AB": {"name": "Annotated By", "d_type": str},
                 "NF": {"name": "No of Frames", "d_type": int},
                 "TS": {"name": "Target Specie", "d_type": str},
                 "TC": {"name": "Target Color", "d_type": str},
                 "OC": {"name": "Objects Count", "d_type": int},
                 "CL": {"name": "Clarity Level", "d_type": int},
                 "SV": {"name": "Scale Var", "d_type": bool},
                 "OV": {"name": "Out of View", "d_type": bool},
                 "PO": {"name": "Partial Occl", "d_type": bool},
                 "FO": {"name": "Full Occl", "d_type": bool},
                 "DF": {"name": "Def", "d_type": bool},
                 "LR": {"name": "Low Res", "d_type": bool},
                 "FM": {"name": "Fast Motion", "d_type": bool},
                 "MB": {"name": "Motion Blur", "d_type": bool},
                 "SD": {"name": "Similar Distractors", "d_type": bool},
                 "CM": {"name": "Camera Motion", "d_type": bool},
                 "IV": {"name": "Illum Var", "d_type": bool},
                 "CF": {"name": "Camouflage", "d_type": bool},
                 "RS": {"name": "Relative Size", "d_type": float},
                 "WC": {"name": "Water Color", "d_type": float},
                 "TR": {"name": "Target Rotates", "d_type": bool},
                 "PT": {"name": "Partial Target Info", "d_type": bool}}
    return attr_maps

def attr_display_name(attr):
    switcher = {"CL": "UW Visibility",
                 "SV": "Intra-target Scale Variation",
                 "OV": "Out of View",
                 "PO": "Partial Occlusion",
                 "FO": "Full Occlusion",
                 "DF": "Deformation",
                 "LR": "Low Resolution",
                 "FM": "Fast Motion",
                 "MB": "Motion Blur",
                 "SD": "Swarm Distractors",
                 "CM": "Camera Motion",
                 "IV": "Illumination Variation",
                 "CF": "Camouflage",
                 "RS": "Relative Target Size Variation",
                 "WC": "Water Color Variation",
                 "TR": "Out of Plane Rotation",
                 "PT": "Partial Target Information"}
    return switcher.get(attr, "nothing")


def water_color(val):
    switcher = {1: "Colorless",
                2: "Ash",
                3: "Green",
                4: "Light Blue",
                5: "Blue",
                6: "Gray",
                7: "Light Green",
                8: "Deep Blue",
                9: "Dark",
                10: "Partly Blue",
                11: "Light Purple",
                12: "Light Yellow",
                13: "Light Brown",
                14: "Blue Black",
                15: "Cyan",
                16: "Gray Blue"}
    return switcher.get(val, "nothing")


def generate_attr_display(attr, opr, val, n_vids):
    attr_full_name = attr_display_name(attr)
    
    display = attr_full_name
    if opr is None:
        if val != 1:
            display = f"No {display}"
    else:
        if opr == "e":
            if attr == "CL" and val==3:
                display = f"{display}-MID"
            elif attr == "CL" and val != 3:
                display = f"{display}-Level {val}"
            elif attr == "WC":
                display = f"{display}-{water_color(val)}"
        elif opr == "l":
            if attr == "CL" and val == 3:
                display = f"{display}-LOW"
            else:
                display = f"{display} $<$ {val}"
        elif opr == "g":
            if attr == "CL" and val == 3:
                display = f"{display}-HiGH"
            else:
                display = f"{display} $>$ {val}"

    return f"{display}: ({n_vids})" 


def get_val_opr(attr_val, attr_dtype):
    if attr_dtype is bool:
        val, opr = int(attr_val), None
    else:
        opr = attr_val.split()[0][0]
        val = attr_val.split()[0][1:] 
        val = float(val) if "." in val else int(val)
    
        assert opr == "e" or opr == "l" or opr == "g"    
    return val, opr 

def copy_anno(src_anno, dest_path, opr=None, req_val=None, atr_val=None, just_copy=False):
    
    if not just_copy:
        atr_val = float(atr_val) if "." in str(atr_val) else int(atr_val)

        if (opr is None and atr_val == req_val) or \
            (opr == "e" and atr_val == req_val) or \
            (opr == "l" and atr_val < req_val) or \
            (opr == "g" and atr_val > req_val):
            just_copy = True
            
    if just_copy:
        if not os.path.exists(dest_path): 
            os.makedirs(dest_path)
        shutil.copy(src_anno, dest_path)
        

def create_attr_json(videos_folder_path, dataset_name, 
                     gt_file_name="groundtruth_rect.txt",
                     convert_region=False, delimiter='\t'):
    '''Creates JSON file for our videos:
    It follows the OTB json creation format'''

    all_folders = glob(videos_folder_path, recursive = True)
    all_folders.sort()

    videos_folders = []

    videos_folders = all_folders

    videos_dicts = {}
    for video_folder in videos_folders:
        gt_rect = []

        if '\\' not in video_folder:
            video_dir = video_folder.split('/')[-2]
        else:
            video_dir = video_folder.split('\\')[-2]

        for file in os.listdir(video_folder):

            # Read GT bboxes in the text file and convert to list of bboxes
            if file.endswith(gt_file_name):   
                with open(os.path.join(video_folder, file)) as f:   # Read all
                    gt_bboxes = f.readlines()
                
                # Initial GT bbox
                init_rect = (gt_bboxes[0].split('\n')[0]).split(delimiter)
                init_rect = [int(x) for x in list(map(float, init_rect))]

                if convert_region:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(init_rect, dtype=np.float32))
                    bb = cxy_wh_2_rect((cx, cy,), (w, h))
                    init_rect = [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])]

                # All GT bboxes
                for i, gt_bb in enumerate(gt_bboxes):
                    gt_bb = (gt_bb.split('\n')[0]).split(delimiter)
                    rect = [int(x) for x in list(map(float, gt_bb))]

                    if convert_region:
                        cx,cy,w,h = get_axis_aligned_bbox(np.array(rect, dtype=np.float32))
                        bb = cxy_wh_2_rect((cx, cy,), (w, h))
                        rect = [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])]

                    gt_rect.append(rect)

        # Create dictionary of the current video
        video_dict = {}
        video_dict["video_dir"] = video_dir
        video_dict["init_rect"] = init_rect
        video_dict["img_names"] = None
        video_dict["gt_rect"] = gt_rect
        video_dict["attr"] = None   # List of Video Attributes

        # Append current video_dict to videos_dicts
        videos_dicts[video_dir] = video_dict
    
    # Save dict as json file
    json_dir = videos_folder_path.split("*/")[0]
    with open(os.path.join(json_dir,f"{dataset_name}.json"), "w") as f:
        json.dump(videos_dicts, f)


class UTBAttrVideo():
    """
    Args:
        name: video name
        root: dataset root
        video_dir: video directory
        init_rect: init rectangle
        img_names: image names
        gt_rect: groundtruth rectangle
        attr: attribute of video
    """
    def __init__(self, name, root, video_dir, init_rect, img_names,
            gt_rect, attr, load_img=False):
        self.name = name
        self.video_dir = video_dir
        self.init_rect = init_rect
        self.gt_traj = gt_rect
        self.attr = attr
        self.pred_trajs = {}
        #self.img_names = [os.path.join(os.path.abspath(root), os.path.abspath(x)) for x in img_names]
        self.img_names = None
        self.imgs = None

    def load_tracker(self, path, tracker_names=None, store=True):
        """
        Args:
            path(str): path to result
            tracker_name(list): name of tracker
        """
        if not tracker_names:
            tracker_names = [x.split('/')[-1] for x in glob(path)
                    if os.path.isdir(x)]
        if isinstance(tracker_names, str):
            tracker_names = [tracker_names]
        for name in tracker_names:
            traj_file = os.path.join(path, name, f'{self.name}.txt')
            if os.path.exists(traj_file):
                with open(traj_file, 'r') as f :
                    pred_traj = [list(map(float, x.strip().split('\t')))
                            for x in f.readlines()]
            else:
                print("File not exists: ", traj_file)

            if store:
                self.pred_trajs[name] = pred_traj
            else:
                return pred_traj

        self.tracker_names = list(self.pred_trajs.keys())


class UTBAttrDataset(Dataset):
    """
    Args:
        name: dataset name
        dataset_root: dataset root
        load_img: wether to load all imgs
    """
    def __init__(self, name, dataset_root, load_img=False):
        super(UTBAttrDataset, self).__init__(name, dataset_root)
        with open(os.path.join(dataset_root, name+".json"), "r") as f:
            meta_data = json.load(f)

        # load videos
        pbar = tqdm(meta_data.keys(), desc='loading '+name, ncols=100)
        self.videos = {}
        for video in pbar:
            pbar.set_postfix_str(video)
            self.videos[video] = UTBAttrVideo(video,
                                          dataset_root,
                                          meta_data[video]['video_dir'],
                                          meta_data[video]['init_rect'],
                                          meta_data[video]['img_names'],
                                          meta_data[video]['gt_rect'],
                                          meta_data[video]['attr'],
                                          load_img)