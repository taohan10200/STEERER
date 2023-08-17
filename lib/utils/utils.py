# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import random
import shutil

def get_world_size():
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()

def get_rank():
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg
class AverageCategoryMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self,num_class):
        self.num_class = num_class
        self.reset()

    def reset(self):
        self.cur_val = np.zeros(self.num_class)
        self.sum = np.zeros(self.num_class)


    def update(self, cur_val):
        self.cur_val = cur_val
        self.sum += cur_val


def random_seed_setting(config):

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    seed = config.seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(config.GPUS).strip('(').strip(')')


def copy_cur_env(work_dir, dst_dir, exception):
    # if os.path.exists(dst_dir):
    #     shutil.rmtree(dst_dir)
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    for filename in os.listdir(work_dir):

        file = os.path.join(work_dir,filename)
        dst_file = os.path.join(dst_dir,filename)

        if os.path.isdir(file) and filename not in exception:
            shutil.copytree(file, dst_file)
        elif os.path.isfile(file):
            shutil.copyfile(file,dst_file)
def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.log_dir)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.dataset.name
    model = cfg.network.backbone+'_'+cfg.network.sub_arch
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)

    if phase=='test':
        test_output_dir = root_output_dir / dataset / phase /cfg_name
        print('=> creating {}'.format(test_output_dir))
        test_output_dir.mkdir(parents=True, exist_ok=True)


        final_log_file = test_output_dir / log_file
        head = '%(asctime)-15s %(message)s'
        logging.basicConfig(filename=str(final_log_file),
                            format=head)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        console = logging.StreamHandler()
        logging.getLogger('').addHandler(console)

        return logger, str(test_output_dir)

    elif phase == 'train':
        resume_path = cfg.train.resume_path

        val_output_dir =  Path(cfg.log_dir) / dataset / model / 'val'
        val_output_dir.mkdir(parents=True, exist_ok=True)

        if resume_path is not None:
            train_log_dir = resume_path
            final_log_file = Path(train_log_dir) / (os.path.basename(train_log_dir)+'_train.log')
        else:
            train_log_dir = Path(cfg.log_dir) / dataset / model / \
                    (cfg_name + '_' + time_str)
            print('=> creating {}'.format(train_log_dir))
            train_log_dir.mkdir(parents=True, exist_ok=True)

            final_log_file = Path(train_log_dir) / log_file
        head = '%(asctime)-15s %(message)s'
        logging.basicConfig(filename=str(final_log_file),
                            format=head)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        console = logging.StreamHandler()
        logging.getLogger('').addHandler(console)

        return logger,  str(train_log_dir)
    else:
        raise ValueError('phase must be "test" or "train"')


def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix

def adjust_learning_rate(optimizer, base_lr, max_iters, 
        cur_iters, power=0.9):
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    optimizer.param_groups[0]['lr'] = lr
    return lr

import torchvision.utils as vutils
import torchvision.transforms as standard_transforms
import cv2
from PIL import Image
import torch.nn.functional as F

def save_results_more(iter, exp_path,img0, pre_map0,gt_map0,pre_cnt, gt_cnt, pre_points=None,gt_points=None):  # , flow):
    # gt_cnt = gt_map0.sum().item()
    # pre_cnt = pre_map0.sum().item()
    pil_to_tensor = standard_transforms.ToTensor()
    tensor_to_pil = standard_transforms.ToPILImage()

    UNIT_H , UNIT_W = img0.size(1), img0.size(2)
    pre_map0 =  F.interpolate(pre_map0.unsqueeze(0),size=(UNIT_H,UNIT_W)).squeeze(0).numpy()
    gt_map0  =  F.interpolate(gt_map0.unsqueeze(0),size=(UNIT_H,UNIT_W)).squeeze(0).numpy()

    tensor = [img0, gt_map0, pre_map0]

    pil_input0 = tensor_to_pil(tensor[0])

    gt_color_map = cv2.applyColorMap((255 * tensor[1] / (tensor[1].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
    pred_color_map = cv2.applyColorMap((255 * tensor[2] / (tensor[2].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)


    # mask_color_map = cv2.applyColorMap((255 * tensor[8]).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
    RGB_R = (255, 0, 0)
    RGB_G = (0, 255, 0)

    BGR_R = (0,0,255)  # BGR
    BGR_G = (0,255,0)  # BGR
    thickness = 3
    lineType = 4
    pil_input0 = np.array(pil_input0)
    # if boxes is not None:
    #     for i, box in enumerate(boxes, 0):
    #         wh_LeftTop = (box[0], box[1])
    #         wh_RightBottom = (box[2], box[3])
    #         # cv2.rectangle(binar_color_map, wh_LeftTop, wh_RightBottom, point_color, thickness, lineType)
    #         cv2.rectangle(pil_input0, wh_LeftTop, wh_RightBottom, point_color, thickness, lineType)
    if pre_points is not None:
        for i, point in enumerate(pre_points, 0):
            # import pdb
            # pdb.set_trace()
            point = point.astype(np.int32)
            point = (point[0], point[1])
            cv2.drawMarker(pil_input0, point, RGB_G, markerType=cv2.MARKER_CROSS,markerSize=15,thickness=3)
            cv2.circle(pred_color_map, point,2, BGR_R,thickness)
            # cv2.drawMarker(pil_input0, point, RGB_R, markerType=cv2.MARKER,markerSize=20,thickness=3)

    if gt_points is not None:
        for i, point in enumerate(gt_points, 0):
            point = point.astype(np.int32)
            point = (point[0], point[1])
            cv2.circle(pil_input0, point,4, RGB_R,thickness)


    cv2.putText(gt_color_map, 'GT:'+str(gt_cnt), (100,150), cv2.FONT_HERSHEY_SIMPLEX,
                5, (255,255,255),thickness=4)
    cv2.putText(pred_color_map, 'Pre:'+str(round(pre_cnt,1)), (100, 150),cv2.FONT_HERSHEY_SIMPLEX,
                5, (255,255,255), thickness=4)
    pil_input0 = Image.fromarray(pil_input0)

    pil_label0 = Image.fromarray(cv2.cvtColor(gt_color_map, cv2.COLOR_BGR2RGB))
    pil_output0 = Image.fromarray(cv2.cvtColor(pred_color_map, cv2.COLOR_BGR2RGB))

    imgs = [pil_input0, pil_label0, pil_output0]

    w_num , h_num=1, 3

    target_shape = (w_num * (UNIT_W + 10), h_num * (UNIT_H + 10))
    target = Image.new('RGB', target_shape)
    count = 0
    for img in imgs:
        x, y = int(count%w_num) * (UNIT_W + 10), int(count // w_num) * (UNIT_H + 10)  # 左上角坐标，从左到右递增
        target.paste(img, (x, y, x + UNIT_W, y + UNIT_H))
        count+=1

    # img = np.vstack([a,b])*255
    # import pdb
    # pdb.set_trace()
    target.save(os.path.join(exp_path,'{}_den.jpg'.format(iter)))
        # cv2.imwrite('./exp/{}_vis_.png'.format(iter), img)
def vis_results(exp_name, writer, img, pred_map, gt_map, binar_map, thresholds, boxes, steps):  # , flow):

    pil_to_tensor = standard_transforms.ToTensor()
    tensor_to_pil = standard_transforms.ToPILImage()
    x = []
    y = []

    for idx, tensor in enumerate(zip(img, pred_map, gt_map, binar_map, thresholds)):
        if idx > 1:  # show only one group
            break

        pil_input = tensor_to_pil(tensor[0])
        pred_color_map = cv2.applyColorMap((255 * tensor[1] / (tensor[1].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        gt_color_map = cv2.applyColorMap((255 * tensor[2] / (tensor[2].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        binar_color_map = cv2.applyColorMap((255 * tensor[3] / (tensor[3].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        thresholds_color_map = cv2.applyColorMap((255 * tensor[4] / (tensor[4].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)



        point_color = (0, 255, 0)  # BGR
        thickness = 1
        lineType = 4
        pil_input = np.array(pil_input)
        # print(pil_input, binar_color_map)
        for i, box in enumerate(boxes, 0):
            wh_LeftTop = (box[0], box[1])
            wh_RightBottom = (box[0] + box[2], box[1] + box[3])
            # print(wh_LeftTop, wh_RightBottom)
            cv2.rectangle(binar_color_map, wh_LeftTop, wh_RightBottom, point_color, thickness, lineType)
            cv2.rectangle(pil_input, wh_LeftTop, wh_RightBottom, point_color, thickness, lineType)

        pil_input = Image.fromarray(pil_input)
        pil_label = Image.fromarray(cv2.cvtColor(gt_color_map, cv2.COLOR_BGR2RGB))
        pil_output = Image.fromarray(cv2.cvtColor(pred_color_map, cv2.COLOR_BGR2RGB))
        pil_binar = Image.fromarray(cv2.cvtColor(binar_color_map, cv2.COLOR_BGR2RGB))
        pil_threshold = Image.fromarray(cv2.cvtColor(thresholds_color_map, cv2.COLOR_BGR2RGB))

        x.extend([pil_to_tensor(pil_input.convert('RGB')), pil_to_tensor(pil_label.convert('RGB')),
                  pil_to_tensor(pil_output.convert('RGB')), pil_to_tensor(pil_binar.convert('RGB')),
                  pil_to_tensor(pil_threshold.convert('RGB'))
                  ])
        # pdb.set_trace()  sum(sum(flow[0].cpu().data.numpy().transpose((1,2,0))[:,:,0]))
        # flow = flow[0].cpu().data.numpy().transpose((1,2,0))
        # flow0 = cv2.applyColorMap((255*flow[:,:,0]/(flow[:,:,0].max()+1e-10)).astype(np.uint8).squeeze(),cv2.COLORMAP_JET)
        # flow1 = cv2.applyColorMap((255*flow[:,:,1]/(flow[:,:,1].max()+1e-10)).astype(np.uint8).squeeze(),cv2.COLORMAP_JET)
        # flow2 = cv2.applyColorMap((255*flow[:,:,2]/(flow[:,:,2].max()+1e-10)).astype(np.uint8).squeeze(),cv2.COLORMAP_JET)
        # flow0 = Image.fromarray(cv2.cvtColor(flow0,cv2.COLOR_BGR2RGB))
        # flow1 = Image.fromarray(cv2.cvtColor(flow1,cv2.COLOR_BGR2RGB))
        # flow2 = Image.fromarray(cv2.cvtColor(flow2,cv2.COLOR_BGR2RGB))
        # y.extend([pil_to_tensor(flow0.convert('RGB')), pil_to_tensor(flow1.convert('RGB')), pil_to_tensor(flow2.convert('RGB'))])

    x = torch.stack(x, 0)
    x = vutils.make_grid(x, nrow=3, padding=5)
    x = (x.numpy() * 255).astype(np.uint8)

    # y = torch.stack(y,0)
    # y = vutils.make_grid(y,nrow=3,padding=5)
    # y = (y.numpy()*255).astype(np.uint8)

    # x = np.concatenate((x,y),axis=1)
    writer.add_image(exp_name, x, steps)