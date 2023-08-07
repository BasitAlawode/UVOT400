from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import warnings
import sys

from sympy import true
warnings.filterwarnings("ignore")

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import argparse

from glob import glob as glob
from tqdm import tqdm
from multiprocessing import Pool
import json
import pandas as pd

from toolkit.evaluation import OPEBenchmark
from basit_codes.ao_benchmark import AOBenchmark
from basit_codes.create_json import create_json
from basit_codes.draw_success_precision import draw_success_precision
from basit_codes.attr_utils import *

base_dir = os.getcwd()

#===== Datasets and Attributes Configuration =====================================
# extract_part if true is used to obtain the performance of part of a base dataset 
extract_part, part_datasets = False, ["UTB400_test", "UTB400_test_en"]
extract_part, part_datasets = True, ["UTB400_test"]

# If extract_part is false, attribute evaluation is performed
# See the excel file and mapping below (line 66)
base_dataset_names = ["UTB400"] # "UTB400_IPT", "UTB400_WN"

# Check attr_utils.py for all possible attributes
# Format examples: if bool SD_0, SD_1... If float, [RS_l0.5, RS_e0.2, RS_g0.6]

attributes = ["CL_e0", "CL_e1", "CL_e2", "CL_e3", "CL_e4", "CL_e5", 
        "SV_0", "SV_1", "OV_0", "OV_1", "PO_0", "PO_1", "FO_0", "FO_1",  
        "DF_0", "DF_1", "LR_0", "LR_1", "FM_0", "FM_1", "MB_0", "MB_1", 
        "SD_0", "SD_1", "CM_0", "CM_1", "IV_1", "IV_0", "CF_0", "CF_1", 
        "TR_0", "TR_1", "PT_0", "PT_1", "RS_l0.25", "RS_g0.25", "RS_l0.5", "RS_g0.5", 
        "WC_e1", "WC_e2", "WC_e3", "WC_e4", "WC_e5", "WC_e6", "WC_e7", "WC_e8", 
        "WC_e9", "WC_e10", "WC_e11", "WC_e12", "WC_e13", "WC_e14", "WC_e15", "WC_e16"]


#==== Trackers =================================================================== 
#trackers = ["ToMP", "RTS", "KeepTrack"]
#trackers = ["ARDiMP", "STMTrack", "AutoMatch"]
#trackers = ["TrDiMP", "TrSiam"]
#trackers = ["TransT", "SiamBAN", "SiamGAT"]
#trackers = ["SparseTT"]

#trackers = ["ToMP", "RTS", "KeepTrack", "ARDiMP", "STMTrack", "AutoMatch", \
#            "TrDiMP", "TrSiam", "TransT", "SiamBAN", "SiamGAT", "SparseTT"]
#trackers = ["ToMP", "RTS", "ARDiMP", "STMTrack", "TrDiMP", "TransT"]

trackers = ["SiamFC", "SiamRPN", "SiamMASK", "SiamBAN", "SiamCAR", "ATOM", "DiMP", \
            "PrDiMP", "STARK", "TrTr", "KeepTrack", "TransT", "TrDiMP", "TrSiam", \
            "ToMP", "SiamGAT", "RTS", "LWL", "SiamAttn", "CSWinTT", "SparseTT", \
            "SiamRPN++-RBO", "ARDiMP", "STMTrack", "AutoMatch"]
#=================================================================================


dataset_names = []

if extract_part:
        for p in part_datasets:
                dataset_names.append([p, p.split("_")[0]])
else:
        for b_data in base_dataset_names:
                for attr in attributes:
                        dataset_names.append([f"{b_data}_{attr}", b_data])

datasets_dir = "testing_datasets" # Testing Datasets dir
trackers_results_dir = "trackers_results"  # Tracker results path 

# Attributes Columns Mapping
if not extract_part:
        attrs_mapping = attr_mapping()
        attr_file = "Annotation_UTB400_all_attr.xlsx"

global num
num = 1
show_video_level, plot_success_precision, norm_precision = False, True, True
show_top = 25   # Top number of trackers to plot
compute_ao = False
legend_cols = 1   # Number of legend columns for display


def main(dataset_name, num): 
        assert len(trackers) > 0
        num = min(num, len(trackers))
        
        dataset_name, base_dataset = dataset_name[0], dataset_name[1]

        dataset_root = os.path.join(base_dir, datasets_dir, dataset_name)
        trackers_results_path = os.path.join(base_dir, trackers_results_dir, dataset_name)
        
        if extract_part:
                # Read partial video files in dataset
                video_nos = [(f.path.split("/")[-1]).split("_")[-1] 
                             for f in os.scandir(dataset_root) if f.is_dir()]
                attr_data = [None]*len(video_nos)
        else:
                # Get attribute name
                attr_key = dataset_name.split("_")[-2]
                attr_val = dataset_name.split("_")[-1]
                
                attr_name = attrs_mapping[attr_key]["name"]
                attr_dtype = attrs_mapping[attr_key]["d_type"]
                
                # Get all attribute data
                all_attrs_data = pd.read_excel(f"{datasets_dir}/{attr_file}",
                                usecols=["Video Number", attr_name],
                                converters={"Video Number":str, 
                                                attr_name:str})
                video_nos = all_attrs_data["Video Number"].tolist()
                attr_data = all_attrs_data[attr_name].tolist()
                
                # Get attribute test value and operation
                req_val, opr = get_val_opr(attr_val, attr_dtype)
        
        # For each video, copy both the true bbox text files 
        for vid_no, atr_val in zip(video_nos, attr_data):
                video_name = f"Video_{vid_no}"
                
                # Destination Directory
                dest_vid_dir = os.path.join(dataset_root, video_name)
                
                # Copy annotations based on attribute and operation
                src_vid_dir = os.path.join(datasets_dir, base_dataset, video_name)
                anno_path = os.path.join(src_vid_dir, "groundtruth_rect.txt")
                
                if extract_part:
                        copy_anno(anno_path, dest_vid_dir, just_copy=True)
                else:
                        copy_anno(anno_path, dest_vid_dir, opr=opr, 
                                  req_val=req_val, atr_val=atr_val)
        
        # Create the attribute JSON file
        if not os.path.exists(os.path.join(dataset_root, f"{dataset_name}.json")):
                print('Dataset JSON does not exit... Attempting to create one.')

                create_json(f"{dataset_root}/*/", dataset_name)
                print("JSON created and saved.")
        
        
        # Copy each tracker result for each video where attribute is true        
        for tracker in trackers:
                tracker_result_dir = os.path.join(trackers_results_path, tracker)
                if not os.path.exists(tracker_result_dir): 
                        os.makedirs(tracker_result_dir)
                        
                # For each video, copy tracker predicted bbox text file 
                src_vid_dir = os.path.join(trackers_results_dir, base_dataset, tracker) 
                for vid_no, atr_val in zip(video_nos, attr_data):
                        video_name = f"Video_{vid_no}"
                        
                        # Copy tracker annotations based on attribute and operation
                        anno_path = os.path.join(src_vid_dir, f"{video_name}.txt")
                        
                        if extract_part:
                                copy_anno(anno_path, tracker_result_dir, just_copy=True)
                        else:
                                copy_anno(anno_path, tracker_result_dir, opr=opr, 
                                          req_val=req_val, atr_val=atr_val)
                
                #print('.....Done.')
                                 
        # Create dataset and set the trackers
        dataset = UTBAttrDataset(dataset_name, dataset_root, load_img=False)
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
        benchmark.show_result(success_ret, precision_ret, norm_precision_ret,
                        show_video_level=show_video_level)

        # Get attribute display legend
        n_vids = len(success_ret[trackers[0]].keys())
        if extract_part:
                attr_display = "ALL" if not "test" in dataset_name.lower() \
                        else f"UW-VOT400 Testing Set ({n_vids})"
                        
        else:
                attr_display = generate_attr_display(attr_key, opr, req_val, n_vids)
                                              
        # Plottings
        if not os.path.exists(os.path.join(trackers_results_path, "plots")):
                os.makedirs(os.path.join(trackers_results_path, "plots"))
        if plot_success_precision:
                tracker_result_dir = os.path.join(trackers_results_path, trackers[0])
                videos_paths = sorted(glob.glob(f"{tracker_result_dir}/*.txt"))
                videos = [(k.split("/")[-1]).split(".")[0] for k in videos_paths]
                #draw_success_precision(success_ret, dataset_name, videos, 'ALL', \
                #         precision_ret=precision_ret, norm_precision_ret=norm_precision_ret, 
                #         show_top=show_top)
                draw_success_precision(success_ret, dataset_name, videos, attr_display,
                         precision_ret=precision_ret, norm_precision_ret=norm_precision_ret, 
                         show_top=show_top, legend_cols=legend_cols)
        
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

        #print('Completed....')
        #sys.stdout = open("terminal_out.txt", "w")
        original_stdout = sys.stdout
        #with open(f'terminal_out_{dataset_name}.txt', 'w') as f:
        #        sys.stdout = f 
        #        sys.stdout = original_stdout  
        

if __name__ == "__main__":
        for dataset_name in dataset_names:
                print(f'\n\n Results for Dataset {dataset_name}\n\n')
                main(dataset_name, num) 
        print('Completed....')        
                     
