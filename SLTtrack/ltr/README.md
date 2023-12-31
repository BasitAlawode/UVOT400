# LTR

A general PyTorch based framework for learning tracking representations. 
## Table of Contents

* [Quick Start](#quick-start)
* [Overview](#overview)
* [Training your own networks](#training-your-own-networks)

## Quick Start
The installation script will automatically generate a local configuration file  "admin/local.py". In case the file was not generated, run ```admin.environment.create_default_local_file()``` to generate it. Next, set the paths to the training workspace, 
i.e. the directory where the checkpoints will be saved. Also set the paths to the datasets you want to use. If all the dependencies have been correctly installed, you can train a network using the run_training.py script in the correct conda environment.  
```bash
conda activate pytracking
python run_training.py train_module train_name
```
Here, ```train_module``` is the sub-module inside ```train_settings``` and ```train_name``` is the name of the train setting file to be used.

For example, you can train using the included default SLT-TransT settings by running:
```bash
python run_training.py slt_transt slt_transt
```


## Overview
The framework consists of the following sub-modules.  
 - [actors](actors): Contains the actor classes for different trainings. The actor class is responsible for passing the input data through the network can calculating losses.  
 - [admin](admin): Includes functions for loading networks, tensorboard etc. and also contains environment settings.  
 - [dataset](dataset): Contains integration of a number of training datasets, namely [TrackingNet](https://tracking-net.org/), [GOT-10k](http://got-10k.aitestunion.com/), [LaSOT](https://cis.temple.edu/lasot/), 
 [ImageNet-VID](http://image-net.org/), [DAVIS](https://davischallenge.org), [YouTube-VOS](https://youtube-vos.org), [MS-COCO](http://cocodataset.org/#home), [SBD](http://home.bharathh.info/pubs/codes/SBD), [LVIS](https://www.lvisdataset.org), [ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html), [MSRA10k](https://mmcheng.net/msra10k), and [HKU-IS](https://sites.google.com/site/ligb86/hkuis). Additionally, it includes modules to generate synthetic videos from image datasets. 
 - [data_specs](data_specs): Information about train/val splits of different datasets.   
 - [data](data): Contains functions for processing data, e.g. loading images, data augmentations, sampling frames from videos.  
 - [external](external): External libraries needed for training. Added as submodules.  
 - [models](models): Contains different layers and network definitions.  
 - [trainers](trainers): The main class which runs the training.  
 - [train_settings](train_settings): Contains settings files, specifying the training of a network.   
 
## Training your own networks
To train a custom network using the toolkit, the following components need to be specified in the train settings. For reference, see [atom.py](train_settings/bbreg/atom.py).  
- Datasets: The datasets to be used for training. A number of standard tracking datasets are already available in ```dataset``` module.  
- Processing: This function should perform the necessary post-processing of the data, e.g. cropping of target region, data augmentations etc.  
- Sampler: Determines how the frames are sampled from a video sequence to form the batches.  
- Network: The network module to be trained.  
- Objective: The training objective.  
- Actor: The trainer passes the training batch to the actor who is responsible for passing the data through the network correctly, and calculating the training loss.  
- Optimizer: Optimizer to be used, e.g. Adam.  
- Trainer: The main class which runs the epochs and saves checkpoints. 
 

 