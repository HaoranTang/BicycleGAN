# 680 Final Project: BicycleGAN
##### Haoran Tang
### Instructions

#### 1. Training
To train the network, please run train.py. Change hyper-parameters and folder paths inside if necessary. This file can produce training visualizations.
### 2. Inference
To conduct the inference, please run inference.py. Change hyper-parameters and folder paths (create before running code to save images) inside if necessary. This file can produce loss curves, visualizations from validation set, images for FID and LPIPS. Please specify each setting by --loss_curves, --random, --fid, --lpips.
### 3. FID and LPIPS scores
To calculate FID and LPIPS, please run compute_fid.sh and comopute_lpips.sh indivisually. Please install dependencies before running and change folder paths if necessary.
### 4. Support files
Model is defined in models.py and dataset class in datasets.py. I adopted download_dataset.sh from the original implementation to download datasets, but this is trivial. I adopted the python file for calculating lpips score with folder paths from the original lpips repository and modified a little bit for my folder structure, and named it as calc_lpips.py. The vis_tools.py is given by default.

### 5. Log files
Performance, visualizations and scores are all reported in the report. If you need checkpoint, .txt for scores, or more images, please let me know, thank you!
