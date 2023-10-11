# Benchmarking Trackers on Custom Videos

To benchmark SOTA trackers on your custom videos:

1. Setup the experiment environment as described [here](README.md/#experiment-environment-setup). 

1. Download the pretrained tracker(s) you want to benchmark on as described [here](1_Benchmarking_README.md/#downloading-pre-trained-trackers-models).

2. Put your videos in the testing_datasets folder. 

The folder structure should look like this:

  ```
   ${PROJECT_ROOT}
    -- testing_datasets
        -- Your_Videos_Parent_Folder
          -- your_video_1_folder
            --imgs
              |-- 0001.jpg
              |-- 0002.jpg
              |-- 0003.jpg
              ...
            -- groundtruth_rect.txt
          -- your_video_2_folder
            --imgs
              |-- 0001.jpg
              |-- 0002.jpg
              |-- 0003.jpg
              ...
            -- groundtruth_rect.txt
          -- your_video_3_folder
            --imgs
              |-- 0001.jpg
              |-- 0002.jpg
              |-- 0003.jpg
              ...
            -- groundtruth_rect.txt
          ...
   ```
NOTE: Each video's groundtruth_rect file should be of the format **N by [x,y,w,h]** where N is the number of sequence in the video (number of rows in the text file), **[x,y], w, h** denote the coordinate of the top-left corner, width and height of the target bounding-box in each frame respectively.

3. Open [main.py](main_eval.py) in your favourite editor. 

4. Change dataset_names list in line 35-36 to your_video_parent_folder name.

5. Also, edit lines 45-48 to reflect the trackers you're interested in.

6. Then, run. 

```bash
python main_eval.py
```

7. Tracking results will be shown in the terminal. Plots will be found in tracking_results/your_video_parent_folder/plot.


## List of Available Trackers

Here is a list of currently available trackers.

1. Discriminative Correlation Filter-based Trackers:
   - [ATOM](https://github.com/visionml/pytracking), [DiMP](https://github.com/visionml/pytracking), [KYS](https://github.com/visionml/pytracking), [PrDiMP](https://github.com/visionml/pytracking), [ARDiMP](https://github.com/MasterBin-IIAU/AlphaRefine)
2. Deep Siamese Trackers
   - [SiamFC](https://github.com/got-10k/siamfc), [SiamRPN](https://github.com/STVIR/pysot), [SiamMask](https://github.com/STVIR/pysot), [SiamCAR](https://github.com/ohhhyeahhh/SiamCAR), [SiamBAN](https://github.com/hqucv/siamban), [SiamGAT](https://github.com/ohhhyeahhh/SiamGAT), [SiamAttn](https://github.com/msight-tech/research-siamattn), [RBO-SiamRPN++](https://github.com/sansanfree/RBO), , [KeepTrack](https://github.com/visionml/pytracking)
3. Transformer-driven Trackers
   - [TrSiam](https://github.com/594422814/TransformerTrack), [TrDiMP](https://github.com/594422814/TransformerTrack), [STMTrack](https://github.com/fzh0917/STMTrack), [TrTr](https://github.com/tongtybj/TrTr), [TransT](https://github.com/chenxin-dlut/TransT), [Stark](https://github.com/researchmm/Stark), [ToMP](https://github.com/visionml/pytracking), [RTS](https://github.com/visionml/pytracking), [CSWinTT](https://github.com/SkyeSong38/CSWinTT), [SparseTT](https://github.com/fzh0917/SparseTT), [AutoMatch](https://github.com/JudasDie/SOTS)

NOTE: 
1. We have pulled the trackers from their respective github repositories.
2. More trackers will be added periodically.


## Citation

If you find our work useful for your research, please consider citing:

```bibtex
@article{Alawode2023,
archivePrefix = {arXiv},
arxivId = {2308.15816},
author = {Alawode, Basit and Dharejo, Fayaz Ali and Ummar, Mehnaz and Guo, Yuhang and Mahmood, Arif and Werghi, Naoufel and Khan, Fahad Shahbaz and Javed, Sajid},
eprint = {2308.15816},
title = {{Improving Underwater Visual Tracking With a Large Scale Dataset and Image Enhancement}},
url = {http://arxiv.org/abs/2308.15816},
volume = {14},
year = {2023}
}

@inproceedings{alawode2022utb180,
  title={UTB180: A High-quality Benchmark for Underwater Tracking},
  author={Alawode, Basit and Guo, Yuhang and Ummar, Mehnaz and Werghi, Naoufel and Dias, Jorge and Mian, Ajmal and Javed, Sajid},
  booktitle={{ACCV}},
  year={2022}
}
```