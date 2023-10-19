# Benchmarking Trackers on Custom Videos

To benchmark SOTA trackers on your custom videos:

1. Setup the experiment environment as described [here](README.md/#experiment-environment-setup). 

2. Download the pretrained tracker(s) you want to benchmark on as described below:

 - See a list of all available trackers [here](1_Custom_Benchmarking_README.md/#list-of-currently-available-trackers).

 - Download the tracker pre-trained model from the link below: 
 
 [Link to Pre-trained Trackers Weights](https://kuacae-my.sharepoint.com/:f:/g/personal/100060517_ku_ac_ae/EiEaxX6XgplLtXsPv95PURUBSNODK-irvo46Jks38-OmjA?e=bF3X59). 
 
 - Put it in the trained_trackers folder.

 - You can add as many trackers as you want from the list of available trackers.
 
As an example, the structure of the trained_trackers folder should be as below:

  ```
   ${PROJECT_ROOT}
    -- trained_trackers
        -- ardimp
            |-- ardim tracker model
        -- automatch
            |-- automatch tracker model
        -- cswintt
            |-- cswintt tracker model
        ...
   ```

NOTE: The pretrained models provided in the link above were gotten from the respective tracker repositories.

3. Put your videos in the testing_datasets folder. 

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

4. Open [main.py](main_eval.py) in your favourite editor. 

5. Change dataset_names list in line 35-36 to your_video_parent_folder name.

6. Also, edit lines 45-48 to reflect the trackers you're interested in.

7. Then, run. 

```bash
python main_eval.py
```

7. Tracking results will be shown in the terminal. Plots will be found in tracking_results/your_video_parent_folder/plot.


## List of Currently Available Trackers

Below is a list of currently available trackers (Will be updated regularly).

<table>
  <tr>
    <th>Venue & Year</th>
    <th>Name & Link</th>
    <th style="border-right:1px solid white"></th>
    <th>Venue & Year</th>
    <th>Name & Link</th>
  </tr>

  <tr>
    <td></td>
    <td><a href=""></a></td>
    <td style="border-right:1px solid white"></td>
    <td>CVPR 2023</td>
    <td><a href="https://github.com/jimmy-dq/DropTrack">DropTrack</a></td>
  </tr>

  <tr>
    <td>CVPR 2023</td>
    <td><a href="https://github.com/Little-Podi/GRM">GRM</a></td>
    <td style="border-right:1px solid white"></td>
    <td>Ocean Eng 2023</td>
    <td><a href="https://github.com/LiYunfengLYF/UOSTrack">UOSTrack</a></td>
  </tr>

  <tr>
    <td>NeurlPS 2023</td>
    <td><a href="https://github.com/MCG-NJU/MixFormerV2">MixFormerV2</a></td>
    <td style="border-right:1px solid white"></td>
    <td>ECCV 2022</td>
    <td><a href="https://github.com/MCG-NJU/MixFormer">MixFormer</a></td>
  </tr>

  <tr>
    <td>ECCV 2022</td>
    <td><a href="https://github.com/byminji/SLTtrack/tree/master">SLT-TransT</a></td>
    <td style="border-right:1px solid white"></td>
    <td>ECCV 2022</td>
    <td><a href="https://github.com/Little-Podi/AiATrack">AiATrack</a></td>
  </tr>

  <tr>
    <td>ECCV 2022</td>
    <td><a href="https://github.com/LPXTT/SimTrack">SimTrack</a></td>
    <td style="border-right:1px solid white"></td>
    <td>ECCV 2022</td>
    <td><a href="https://github.com/botaoye/OSTrack">OSTrack</a></td>
  </tr>

  <tr>
    <td>CVPR 2022</td>
    <td><a href="https://github.com/visionml/pytracking">ToMP</a></td>
    <td style="border-right:1px solid white"></td>
    <td>ECCV 2022</td>
    <td><a href="https://github.com/visionml/pytracking">RTS</a></td>
  </tr>

  <tr>
    <td>ICCV 2022</td>
    <td><a href="https://github.com/SkyeSong38/CSWinTT">CSWinTT</a></td>
    <td style="border-right:1px solid white"></td>
    <td>IJCAI 2022</td>
    <td><a href="https://github.com/fzh0917/SparseTT">SparseTT</a></td>
  </tr>

  <tr>
    <td>ICCV 2022</td>
    <td><a href="https://github.com/sansanfree/RBO">RBO-SiamRPN++</a></td>
    <td style="border-right:1px solid white"></td>
    <td>CVPR 2021</td>
    <td><a href="https://github.com/MasterBin-IIAU/AlphaRefine">ARDiMP</a></td>
  </tr>

  <tr>
    <td>ICCV 2021</td>
    <td><a href="https://github.com/JudasDie/SOTS">AutoMatch</a></td>
    <td style="border-right:1px solid white"></td>
    <td>CVPR 2021</td>
    <td><a href="https://github.com/fzh0917/STMTrack">STMTrack</a></td>
  </tr>

  <tr>
    <td>ICCV 2021</td>
    <td><a href="https://github.com/ohhhyeahhh/SiamGAT">SiamGAT</a></td>
    <td style="border-right:1px solid white"></td>
    <td>ICCV 2021</td>
    <td><a href="https://github.com/visionml/pytracking">KeepTrack</a></td>
  </tr>

  <tr>
    <td>CVPR 2021</td>
    <td><a href="https://github.com/chenxin-dlut/TransT">TransT</a></td>
    <td style="border-right:1px solid white"></td>
    <td>CVPR 2021</td>
    <td><a href="https://github.com/tongtybj/TrTr">TrTr</a></td>
  </tr>

  <tr>
    <td>CVPR 2021</td>
    <td><a href="https://github.com/594422814/TransformerTrack">TrSiam</a></td>
    <td style="border-right:1px solid white"></td>
    <td>CVPR 2021</td>
    <td><a href="https://github.com/594422814/TransformerTrack">TrDiMP</a></td>
  </tr>

  <tr>
    <td>ICCV 2021</td>
    <td><a href="https://github.com/researchmm/Stark">STARK</a></td>
    <td style="border-right:1px solid white"></td>
    <td>CVPR 2020</td>
    <td><a href="https://github.com/visionml/pytracking">PrDiMP</a></td>
  </tr>

  <tr>
    <td>CVPR 2020</td>
    <td><a href="https://github.com/hqucv/siamban">SiamBAN</a></td>
    <td style="border-right:1px solid white"></td>
    <td>CVPR 2020</td>
    <td><a href="https://github.com/ohhhyeahhh/SiamCAR">SiamCAR</a></td>
  </tr>

  <tr>
    <td>ECCV 2020</td>
    <td><a href="https://github.com/visionml/pytracking">KYS</a></td>
    <td style="border-right:1px solid white"></td>
    <td>ECCV 2020</td>
    <td><a href="https://github.com/visionml/pytracking">LWL</a></td>
  </tr>

  <tr>
    <td>CVPR 2020</td>
    <td><a href="https://github.com/visionml/pytracking">TrSiam</a></td>
    <td style="border-right:1px solid white"></td>
    <td>CVPR 2020</td>
    <td><a href="https://github.com/visionml/pytracking">TrDiMP</a></td>
  </tr>
  <tr>
    <td>CVPR 2019</td>
    <td><a href="https://github.com/STVIR/pysot">SiamRPN</a></td>
    <td style="border-right:1px solid white"></td>
    <td>CVPR 2020</td>
    <td><a href="https://github.com/msight-tech/research-siamattn">SiamAttn</a></td>
  </tr>
  <tr>
    <td>ICCV 2019</td>
    <td><a href="https://github.com/visionml/pytracking">DiMP</a></td>
    <td style="border-right:1px solid white"></td>
    <td>CVPR 2019</td>
    <td><a href="https://github.com/visionml/pytracking">ATOM</a></td>
  </tr>
  <tr>
    <td>ECCV 2016</td>
    <td><a href="https://github.com/got-10k/siamfc">SiamFC</a></td>
    <td style="border-right:1px solid white"></td>
    <td>CVPR 2019</td>
    <td><a href="https://github.com/STVIR/pysot">SiamMASK</a></td>
  </tr>
</table>

NOTE: We have pulled the trackers from their respective github repositories.


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