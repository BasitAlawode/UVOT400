# Improving Underwater Visual Tracking With a Large Scale Dataset and Image Enhancement

 - This repository is the official implementation of our work on **Improving Underwater Visual Tracking With a Large Scale Dataset and Image Enhancement**.

[Paper ArXiv Link](https://arxiv.org/abs/2308.15816) | [Dataset](./README.md/#our-dataset-uvot400) | [Thumbnails](https://kuacae-my.sharepoint.com/:f:/g/personal/100060517_ku_ac_ae/EoF4-F76hNhFqvxJfomF6YoBw0Dz5mRYubL9KpoWs6gAkw?e=hzKfxp)


<table>
  <tr>
    <th><div align="center">
  <img src="images/Video2.gif" width="300px" />
  <p>Turtle.</p>
</div></th>
    <th><div align="center">
  <img src="images/Video5.gif" width="300px" />
  <p>Diver.</p>
</div></th>
  </tr>
</table>

![Tracking Sample](images/sample_tracking.png)

## Benchmarking SOTA Trackers on Custom Videos

 - This repository also allows you to quickly benchmark SOTA trackers on your custom videos [see here](1_Custom_Benchmarking_README.md). 


## News:
- Oct. 12, 2023: Added two new trackers [see here](1_Custom_Benchmarking_README.md).
- Oct. 05, 2023: Quickly Benchmark SOTA trackers on your custom videos. 
- Aug. 31, 2023: ArXiv Link to paper provided
- Aug. 30, 2023: Paper submitted for publication
- Aug. 07, 2023: Repository made public.
- June 30, 2023: Dataset (Train and Test Set link available)

### TODO
- [x] Include all pulled trackers folders
- [x] Provide link to download dataset and annotations ([here](README.md/#links-to-datasets))
- [x] Experiment environment creation ([here](README.md/#experiment-environment-setup))
- [x] Pretrained trackers benchmarking results ([here](1_Benchmarking_README.md))
- [x] Reproducing our results ([here](1_Benchmarking_README.md/#reproducing-our-results))
- [x] Provide link to download pretrained trackers models ([here](https://kuacae-my.sharepoint.com/:f:/g/personal/100060517_ku_ac_ae/EiEaxX6XgplLtXsPv95PURUBSNODK-irvo46Jks38-OmjA?e=bF3X59))
- [x] Provide link to paper ([here](https://arxiv.org/abs/2308.15816))
- [x] Provide link to attributes file ([here](https://kuacae-my.sharepoint.com/:x:/g/personal/100060517_ku_ac_ae/EbL0y6an-EhMqLbDtvBSozEBSGjKg__lvevgKSA1udx3jg?e=Q3Sa9B)).
- [x] Benchmarking trackers on custom videos [here](1_Custom_Benchmarking_README.md)
- [x] Provide dataset thumbnails [here](https://kuacae-my.sharepoint.com/:f:/g/personal/100060517_ku_ac_ae/EoF4-F76hNhFqvxJfomF6YoBw0Dz5mRYubL9KpoWs6gAkw?e=hzKfxp). 
- [ ] Attribute-wise evaluation results [here](1_Benchmarking_README.md/#attribute-wise-performance-evaluation)
- [ ] Fine-tuned trackers benchmarking results
- [ ] Enhanced frames trackers benchmarking results

## Our Main Contributions

1. A large and diverse high-quality UVOT400 benchmark dataset is presented, consisting of 400 sequences and 275,000 manually annotated bounding-box frames, introducing 17 distinct tracking attributes with diverse underwater creatures as targets.

2. A large-scale benchmarking of 24 recent SOTA trackers is performed on the proposed dataset, adopting established performance metrics.

3. An UWIE-TR algorithm is introduced. It improves the UVOT performance of SOTA open-air trackers on underwater sequences.

4. The selected SOTA trackers are re-trained on the enhanced version of the proposed dataset resulting in significant performance improvement across all compared trackers.

## Our Dataset: UVOT400

Details about the data collection, annotations, domain-specific tracking attributes can be found in [our paper](https://arxiv.org/abs/2308.15816).

### Links to Datasets

1. Our UVOT400 dataset:
   - Train Set: [Download link](https://kuacae-my.sharepoint.com/:f:/g/personal/100060517_ku_ac_ae/Em9CQUgLcY1BnEBqpGUTrxQBnVPzBfpfPcUW7RmH3EON9w?e=pjNgIY) 
   - Test Set: [Download link](https://kuacae-my.sharepoint.com/:f:/g/personal/100060517_ku_ac_ae/EmJKcYONDL9Kll9OJkArN-4B9UgfBPejZ8LHAxn6aP__Xg?e=21ELVO)
   - Attributes file: [Download link](https://kuacae-my.sharepoint.com/:x:/g/personal/100060517_ku_ac_ae/EbL0y6an-EhMqLbDtvBSozEBSGjKg__lvevgKSA1udx3jg?e=Q3Sa9B). 

2. Our Previous UTB180 Dataset: 
   - [Kaggle Link](https://www.kaggle.com/dataset/21f6e3008c9ac0f51479b93fe0bb0b015469d70153f8987d9f2c3bb3eebbba99). 
   - **Reference:** B. Alawode, Y. Guo, M. Ummar, N. Werghi, J. Dias, A. Mian, and S. Javed, "UTB180: A high-quality benchmark for underwater tracking," in ACCV, 2022.

## Evaluated Trackers

We have utilized several SOTA trackers for the several experiments we have performed. Links to the github repositories of the trackers are as below (click on the tracker name to go to the github page):

1. Discriminative Correlation Filter-based Trackers:
   - [ATOM](https://github.com/visionml/pytracking), [DiMP](https://github.com/visionml/pytracking), [KYS](https://github.com/visionml/pytracking), [PrDiMP](https://github.com/visionml/pytracking), [ARDiMP](https://github.com/MasterBin-IIAU/AlphaRefine)
2. Deep Siamese Trackers
   - [SiamFC](https://github.com/got-10k/siamfc), [SiamRPN](https://github.com/STVIR/pysot), [SiamMask](https://github.com/STVIR/pysot), [SiamCAR](https://github.com/ohhhyeahhh/SiamCAR), [SiamBAN](https://github.com/hqucv/siamban), [SiamGAT](https://github.com/ohhhyeahhh/SiamGAT), [SiamAttn](https://github.com/msight-tech/research-siamattn), [RBO-SiamRPN++](https://github.com/sansanfree/RBO), , [KeepTrack](https://github.com/visionml/pytracking)
3. Transformer-driven Trackers
   - [TrSiam](https://github.com/594422814/TransformerTrack), [TrDiMP](https://github.com/594422814/TransformerTrack), [STMTrack](https://github.com/fzh0917/STMTrack), [TrTr](https://github.com/tongtybj/TrTr), [TransT](https://github.com/chenxin-dlut/TransT), [Stark](https://github.com/researchmm/Stark), [ToMP](https://github.com/visionml/pytracking), [RTS](https://github.com/visionml/pytracking), [CSWinTT](https://github.com/SkyeSong38/CSWinTT), [SparseTT](https://github.com/fzh0917/SparseTT), [AutoMatch](https://github.com/JudasDie/SOTS)

For our work, we have pulled the trackers from their respective github repositories.

## Experiment Environment Setup

1. Create the python environment

```bash
conda create -y --name uvot400 python==3.7.16
conda activate uvot400  
``` 

2. Install pytorch and torchvision
```bash
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

3. Install other packages

```bash
pip install -r requirements.txt
```

4. Build region (for [Pysot](https://github.com/STVIR/pysot) library)
```bash
python setup.py build_ext --inplace
```

## Experiments

For our experiments, we have utilized the success, precision, and normalized precision VOT tracker evaluation metrics. For comparison with GOT10k open-air dataset, the average overlap (AO), success rate 0.50, and 0.75 are utilized.

1. [Protocol I: Benchmarking Pre-trained trackers on UVOT400](1_Benchmarking_README.md)

2. [Protocol II: Finetuning and re-benchmarking](2_Finetuning_Benchmarking.MD)

3. [Protocol III: Image enhancement before tracking](3_Enhanced_Benchmarking_README.md)

## Aknowledgements

- Thanks to the authors of the trackers for providing the implementations.
- Thanks to the [Pysot](https://github.com/STVIR/pysot) and [Pytracking](https://github.com/visionml/pytracking) libraries for providing the tracking evaluation codes.
- This work acknowledges the support provided by the Khalifa University of Science and Technology under Faculty Start-Up grants FSU-2022-003 Award No. 8474000401.

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
