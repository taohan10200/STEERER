# STEERER for  Object Counting and Localizaiotion (ICCV 2023)
## Introduction
This is the official PyTorch implementation of paper: [**STEERER: Resolving Scale Variations for Counting and Localization via Selective Inheritance Learning**](https://arxiv.org/abs/2203.12335), which effectively addressed the issue of scale variations for object counting and localizaioion, demonstrating the state-of-arts counting and localizaiton performance for different catagories, such as crowd,vehicle, crops and trees. 

<!-- ![framework](./figures/framework1.png) -->

# Catalog
- [x] Training and Testing Code (2023.08.18)
- [x] Pretrained models (continuously update)


# Getting started 

## preparatoin 

- **Clone this repo** in the directory (```root/```):


```bash
cd $root
git clone https://github.com/taohan10200/STEERER.git
```
- **Install dependencies.** We use python 3.9 and pytorch >= 1.12.0 : http://pytorch.org.

```bash
conda create -n STEERER python=3.9 -y
conda activate STEERER
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch
cd ${STEERER}
pip install -r requirements.txt
```

- **Prepare datasets and weights.** Pretrained models are available at the [OneDrive](https://connectpolyu-my.sharepoint.com/:f:/g/personal/23040302r_connect_polyu_hk/ErX-l0MtTWtJmSilWuxUrOgBMRYqDSbYuAdoi6a-9Jtlmg?e=OdyvTs) net disk,  The shared  should be look like this, and you can selectively dolownd the files that you want to train or inference. Before starting your training and testing, you should organiza your project as the following directory tree. 

````bash

  $root/
  ├── ProcessedData
  │   ├── SHHB
  │   │   ├── images     # the input images
  │   │   ├── jsons      # the annotated labels
  │   │   ├── train.txt   # the image name of train set 
  │   │   ├── test.txt    # the image name of test set
  │   │   ├── test_gt_loc.txt  # the localization labels for evaluation
  │   │   └──train_gt_loc.txt  # the localization labels for train set (not used)
  │   ├── SHHA
  │   ├── NWPU
  │   ├── QNRF
  │   ├── JHU
  │   ├── MTC
  │   ├── JHU
  │   ├── JHUTRANCOS_v3
  │   └── TREE
  ├── PretrainedModels
  └── STEERER

````

## Training
we provide simplify script to run distributed or cluster training,
```bash
# $1 is the configuration file, $2 is the GPU_ID, also support multiple GPUs, like 1,2,3,4 
sh train.sh configs/SHHB_final.py 1  

```
or if you are trainging on the computer cluster, you could be run

```bash
# $3 the configuration file, $4 is the number of GPUs
sh slurm_train.sh partition_name job_name configs/SHHB_final.py 1
```
 

## Testing
To reproduce the performance, run the similry command like training,

```bash
# $1 is the configuration file, $2 is the checkpoint path, $3 is the GPU_ID, only support single GPU. 
sh test.sh configs/SHHB_final.py PretrainedModels/SHHB.pth 1

```
or if you are trainging on the computer cluster, you could be run

```bash
# $3 the configuration file, $4 is the number of GPUs
sh slurm_train.sh partition_name job_name configs/SHHB_final.py 1
```
 

## Reproduce Performance: TODO 
The results on HT21 and SenseCrowd.

- HT21 dataset

|   Method   |  CroHD11~CroHD15    |  MAE/MSE/MRAE(%)  |
|------------|-------- |-------|
| Paper:  VGG+FPN [2,3]| 164.6/1075.5/752.8/784.5/382.3|141.1/192.3/27.4|
| This Repo's Reproduction:  VGG+FPN [2,3]|138.4/1017.5/623.9/659.8/348.5|160.7/217.3/25.1| 

- SenseCrowd dataset

|   Method   |  MAE/MSE/MRAE(%)|  MIAE/MOAE | D0~D4 (for MAE)  |
|------------|---------|-------|-------|
| Paper:  VGG+FPN [2,3]| 12.3/24.7/12.7 |1.98/2.01 |4.1/8.0/23.3/50.0/77.0| 
| This Repo's Reproduction:  VGG+FPN [2,3] |  11.7/24.6/11.7 | 1.99/1.88| 3.6/6.8/22.4/42.6/85.2 |

# Video Demo
Please visit [bilibili](https://www.bilibili.com/video/BV1cY411H7hr/) or [YouTube]() to watch the video demonstration.
![demo](./figures/demo_screen1.png)
# References
1. Acquisition of Localization Confidence for Accurate Object Detection, ECCV, 2018.
2. Very Deep Convolutional Networks for Large-scale Image Recognition, arXiv, 2014.
3. Feature Pyramid Networks for Object Detection, CVPR, 2017. 

# Citation
If you find this project is useful for your research, please cite:

```
@article{haniccvsteerer,
  title={STEERER: Resolving Scale Variations for Counting and Localization via Selective Inheritance Learning},
  author={Han, Tao, Bai Lei, Lingbo Liu, and Ouyang  Wanli},
  booktitle={ICCV},
  year={2023}
}
```

# Acknowledgement
The released PyTorch training script borrows some codes from the [C^3 Framework](https://github.com/gjy3035/C-3-Framework) and [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork) repositories. If you think this repo is helpful for your research, please consider cite them. 
