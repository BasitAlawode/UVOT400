from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import warnings
import sys

from sympy import true
warnings.filterwarnings("ignore")

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import argparse

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
import json

#from toolkit.evaluation import OPEBenchmark
from basit_codes.ope_benchmark import OPEBenchmark
from basit_codes.ao_benchmark import AOBenchmark

from basit_codes.utb import UTBDataset
from basit_codes.create_json import create_json
from basit_codes.track_video import track_video

#from toolkit.visualization import draw_success_precision
from basit_codes.draw_success_precision import draw_success_precision
from basit_codes.utils import get_trackers_fps

base_dir = os.getcwd()

dataset_names = ["UTB400", "UTB400_test"]  # Whole dataset or Test dataset only

datasets_dir = "testing_datasets" # Testing Datasets dir
trackers_results_dir = "trackers_results"  # Tracker results dir 
trackers_time_dir = "trackers_times" # Tracker tracking time dir 

#====== Attribute Evaluation =========
#dataset_names = ["UTB180", "UOT100", "UTB_clear", "UTB_unclear", "UTB_DF", "UTB_FM",
#        "UTB_FO", "UTB_LR", "UTB_MB", "UTB_OV", "UTB_PO", "UTB_SO", "UTB_SV"]

datasets_dir = "testing_datasets" # Testing Datasets dir
trackers_results_dir = "trackers_results"  # Tracker results path 

trackers = ["SiamFC", "SiamRPN", "SiamMASK", "SiamBAN", "SiamCAR", "ATOM", "DiMP",
            "PrDiMP", "STARK", "TrTr", "KeepTrack", "TransT", "TrDiMP", "TrSiam",
            "ToMP", "SiamGAT", "RTS", "LWL", "SiamAttn", "CSWinTT", "SparseTT",
            "SiamRPN++-RBO", "ARDiMP", "STMTrack", "AutoMatch", "OSTrack", "UOSTrack",
            "GRM", "SimTrack"]
#=================================================================================

num = 1
show_video_level, plot_success_precision, norm_precision = False, True, True
show_top = 25   # Top number of trackers to plot
show_fps = True  # Display trackers number of frames per second
legend_cols = 1   # Number of legend columns for display
compute_ao = False
save_excel = False

def main(base_dir, datasets_dir, dataset_name, trackers_results_dir, trackers, num, show_video_level): 
        assert len(trackers) > 0
        num = min(num, len(trackers))

        trackers_results_path = os.path.join(base_dir, trackers_results_dir, dataset_name)
        trackers_time_path = os.path.join(base_dir, trackers_time_dir, dataset_name)
        dataset_root = os.path.join(base_dir, datasets_dir, dataset_name)

        # Create JSON file for the dataset if it does not exist
        if not os.path.exists(os.path.join(dataset_root, f"{dataset_name}.json")):
                print('Dataset JSON does not exit... Attempting to create one.')

                if dataset_name.lower().startswith("vot"):
                        create_json(f"{dataset_root}/*/", dataset_name, gt_file_name="groundtruth.txt", \
                                convert_region=true, delimiter=',', frames_folder_name="color")
                else:
                        create_json(f"{dataset_root}/*/", dataset_name)
                print("JSON created and saved.")

        #Check if trackers results have been obtained, otherwise obtain it.
        with open(os.path.join(dataset_root, dataset_name+".json"), "r") as f:
            dataset_json = json.load(f)
        
        for tracker in trackers:
                if not os.path.exists(os.path.join(trackers_results_path, tracker)):
                        os.makedirs(os.path.join(trackers_results_path, tracker))
                        os.makedirs(os.path.join(trackers_time_path, tracker))
                        
                for i, video_name in enumerate(dataset_json.keys()):
                        pred_bbox_path = os.path.join(trackers_results_path, tracker, f"{video_name}.txt")
                        track_time_path = os.path.join(trackers_time_path, tracker, f"{video_name}_time.txt")
                        if not os.path.exists(pred_bbox_path):
                                #Run tracker for the video and save it in tracker result directory
                                print(f'{tracker} tracker results for {video_name} ({i+1}/{len(dataset_json.keys())}) of {dataset_name} does not exist')
                                print('...Running tracker on the video frames now... Please wait...')
                                video_details = dataset_json[video_name]
                                
                                # Obtain list of bounding boxes
                                pred_bboxes, track_time = track_video(video_details, tracker, base_dir, dataset_name)
                                with open(pred_bbox_path, "w") as f:    # Save bounding boxes
                                        for pred_bbox in pred_bboxes:
                                                f.write(f"{pred_bbox[0]}\t{pred_bbox[1]}\t{pred_bbox[2]}\t{pred_bbox[3]}\n")
                                with open(track_time_path, "w") as f:    # Save tracking time
                                        for t in track_time:
                                                f.write(f"{t}\n")
                                print('.....Done.')
                                 
        # Create dataset and set the trackers
        dataset = UTBDataset(dataset_name, dataset_root, load_img=False)
        dataset.set_tracker(trackers_results_path, trackers)

        # Benchmarking
        benchmark = OPEBenchmark(dataset)

        #sys.stdout = open("terminal_out.txt", "w")
        success_ret = {}        # Success evaluation
        with Pool(processes=num) as pool:
                for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                trackers), desc='eval success', total=len(trackers), ncols=100):
                        success_ret.update(ret)

        precision_ret = {}      # Precision evaluation
        with Pool(processes=num) as pool:
                for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                trackers), desc='eval precision', total=len(trackers), ncols=100):
                        precision_ret.update(ret)
        
        norm_precision_ret = None
        if norm_precision:
                norm_precision_ret = {} # Norm precision evaluation
                with Pool(processes=num) as pool:
                        for ret in tqdm(pool.imap_unordered(benchmark.eval_norm_precision,
                        trackers), desc='eval norm precision', total=len(trackers), ncols=100):
                                norm_precision_ret.update(ret)

        # Show results
        fps_ret = None
        if show_fps:
                fps_ret = get_trackers_fps(trackers, trackers_time_dir, dataset_name)
        benchmark.show_result(success_ret, precision_ret, 
                              norm_precision_ret, fps_ret=fps_ret, 
                              show_video_level=show_video_level)

        attr_display = "UVOT400" if dataset_name == "UTB400" else "ALL"
        # Plottings
        if not os.path.exists(os.path.join(trackers_results_path, "plots")):
                os.makedirs(os.path.join(trackers_results_path, "plots"))
        if plot_success_precision:
                videos = [k for k in dataset_json.keys()]
                draw_success_precision(success_ret, dataset_name, videos, 
                                       attr_display, precision_ret=precision_ret, 
                                       norm_precision_ret=norm_precision_ret, 
                                       show_top=show_top, 
                                       legend_cols=legend_cols)
        # Save result to excel
        if save_excel:
                benchmark.save_to_excel(success_ret, 
                                        f"{dataset_name}_success_result.xlsx")
        
        if compute_ao:
                benchmark = AOBenchmark(dataset)
                # Eval AO, SR0.5, and SR0.75
                ao_ret = {}        # ao evaluation
                with Pool(processes=num) as pool:
                        for ret in tqdm(pool.imap_unordered(benchmark.eval_ao,
                        trackers), desc='eval AO', total=len(trackers), ncols=100):
                                ao_ret.update(ret)

                sr50_ret = {}      # SR50 evaluation
                with Pool(processes=num) as pool:
                        for ret in tqdm(pool.imap_unordered(benchmark.eval_sr50,
                        trackers), desc='eval SR0.50', total=len(trackers), ncols=100):
                                sr50_ret.update(ret)
                
                sr75_ret = {}  # SR75 evaluation
                with Pool(processes=num) as pool:
                        for ret in tqdm(pool.imap_unordered(benchmark.eval_sr75,
                        trackers), desc='eval SR0.75', total=len(trackers), ncols=100):
                                sr75_ret.update(ret)

                # Show results
                benchmark.show_result(ao_ret, sr50_ret, sr75_ret,
                                        show_video_level=show_video_level)

        print('Completed....')
        #sys.stdout = open("terminal_out.txt", "w")
        original_stdout = sys.stdout
        #with open(f'terminal_out_{dataset_name}.txt', 'w') as f:
        #        sys.stdout = f 
        #        sys.stdout = original_stdout  
        

if __name__ == "__main__":
        for dataset_name in dataset_names:
                print(f'\n\n Results for Dataset {dataset_name}\n\n')
                main(base_dir, datasets_dir, dataset_name, trackers_results_dir, \
                        trackers, num, show_video_level) 
                
                     
