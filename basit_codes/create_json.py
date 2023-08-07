from glob import glob
import json
import os
import numpy as np

from pysot.utils.bbox import get_axis_aligned_bbox, cxy_wh_2_rect

dataset_name = "B100"
codes_dir = os.getcwd()
videos_path = f"{codes_dir}/testing_dataset/{dataset_name}/*/"

def create_json(videos_folder_path, dataset_name, gt_file_name="groundtruth_rect.txt", \
    frames_folder_name="imgs", convert_region=False, delimiter='\t', return_json=False):
    '''Creates JSON file for our videos:
    It follows the OTB json creation format'''

    all_folders = glob(videos_folder_path, recursive = True)
    all_folders.sort()

    videos_folders = []

    #for x in all_folders:   # Select only annotated videos
    #    videos_folders.append(x) if "Video" in x else None
    videos_folders = all_folders

    videos_dicts = {}
    for video_folder in videos_folders:
        gt_rect = []
        img_names = []

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
                
            # Read all saved video frames as images
            if file.find(f"{frames_folder_name}") != -1:
                all_imgs = os.listdir(os.path.join(video_folder, file)) 
                all_imgs.sort()
                for img in all_imgs:
                    img_names.append(f"{video_dir}/{frames_folder_name}/{img}")
                #print('done')

        # Create dictionary of the current video
        video_dict = {}
        video_dict["video_dir"] = video_dir
        video_dict["init_rect"] = init_rect
        video_dict["img_names"] = img_names
        video_dict["gt_rect"] = gt_rect
        video_dict["attr"] = None   # List of Video Attributes

        # Append current video_dict to videos_dicts
        videos_dicts[video_dir] = video_dict
    
    # Save dict as json file
    json_dir = videos_folder_path.split("*/")[0]
    with open(os.path.join(json_dir,f"{dataset_name}.json"), "w") as f:
        json.dump(videos_dicts, f)

    return json_file if return_json else None

if __name__ == '__main__':
    json_file = create_json(videos_path, dataset_name)
    print('Done')