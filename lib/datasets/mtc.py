
import os

import cv2
import numpy as np
from PIL import Image
import json
import torch
from torch.nn import functional as F
import random
from .base_dataset import BaseDataset
from .nwpu import NWPU

class MTC(NWPU):
    def __init__(self,
                 root,
                 list_path,
                 num_samples=None,
                 num_classes=1,
                 multi_scale=True,
                 flip=True,
                 ignore_label=-1,
                 base_size=2048,
                 crop_size=(512, 1024),
                 min_unit = (32,32),
                 center_crop_test=False,
                 downsample_rate=1,
                 scale_factor=(0.5, 1/0.5),
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):

        super(MTC, self).__init__(
            root,
            list_path,
            num_samples,
            num_classes,
            multi_scale,
            flip,
            ignore_label,
            base_size,
            crop_size,
            min_unit ,
            center_crop_test,
            downsample_rate,
            scale_factor,
            mean,
            std)
    # def gen_sample(self, image, points,
    #                multi_scale=True, is_flip=True, center_crop_test=False):
    #
    #     if multi_scale:
    #         scale_factor = 0.5 + random.randint(0, self.scale_factor) / 10.0
    #         a = np.arange(0.7,1, 0.01)
    #         b = np.arange(1,1.5, 0.015)
    #         scale_factor= random.choice(np.concatenate([a,b],0))
    #         # scale_factor = random.uniform(self.rate_range[0], self.rate_range[1])
    #         # a = np.arange(0.25,1,0.05)
    #         # scale_factor= random.choice(np.concatenate([a,1/a],0))
    #
    #         image, points = self.crop_then_scale(image, points, scale_factor)
    #
    #
    #
    #
    #     image = self.input_transform(image)
    #     label = self.label_transform(points, image.shape[:2])
    #
    #     image = image.transpose((2, 0, 1))
    #
    #     if is_flip:
    #         flip = np.random.choice(2) * 2 - 1
    #         image = image[:, :, ::flip]
    #         for i in range(len(label)):
    #             label[i] = label[i][:, ::flip].copy()
    #
    #
    #     return image, label
    def read_files(self):
        files = []
        for item in self.img_list:
            image_id = item[0]
            files.append({
                "img": 'imgs/' + image_id + '.jpg',
                "label": 'jsons/' + image_id + '.json',
                "name": image_id,
            })
        # import pdb
        # pdb.set_trace()
        return files
