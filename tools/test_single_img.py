import os
import pprint
# import _init_paths
from lib.core.Counter import Counter
from lib.utils.utils import create_logger, random_seed_setting
from lib.utils.modelsummary import get_model_summary
from lib.core.cc_function import test_cc
# import datasets
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch
import numpy as np
import timeit
import time
import logging
import argparse
from lib.models.build_counter import Baseline_Counter
from lib.utils.dist_utils import (
    get_dist_info,
    init_dist)
from mmcv import Config, DictAction
import cv2
import torchvision.transforms as T
from PIL import Image

def get_count(img: np.array) -> float:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = Config.fromfile('configs/SHHB_final.py')
    model = Baseline_Counter(config.network, config.dataset.den_factor, config.train.route_size, device)
    pretrained_dict = torch.load('PretrainedModels/SHHB_mae_5.8_mse_8.5.pth')
    model.load_state_dict(pretrained_dict,strict=True)
    model = model.to(device)
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((1536, 2048)),
        T.ToTensor(),
    ])
    img = transform(img)
    img = img.unsqueeze(0).to(device)
    labels = list()
    labels.append(torch.zeros(1, 1536, 2048))
    labels.append(torch.zeros(1, 768, 1024))
    labels.append(torch.zeros(1, 384, 512))
    labels.append(torch.zeros(1, 192, 256))
    for i in range(len(labels)):
        labels[i] = labels[i].to(device)
    with torch.no_grad():
        result = model(img, labels=labels, mode='val')
    pre_den = result['pre_den']['1']
    pred_cnt = pre_den.sum().item()
    return pred_cnt

if __name__ == '__main__':
    imgs = [
        
    ]
    for i in imgs:
        img = cv2.imread(i)
        cnt = get_count(img)
        print(cnt)