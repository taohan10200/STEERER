# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------
import os
import pprint
import sys
import _init_paths
from lib.core.Counter import Counter
from lib.utils.utils import create_logger, random_seed_setting
from lib.utils.modelsummary import get_model_summary
from lib.core.cc_function import test_cc, test
import datasets
import lib_cls.models as models
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch
import numpy as np
import timeit
import time
import logging
import argparse
from lib.models.build_counter import Baseline_Counter
from lib.utils.utils import *
from tqdm import tqdm
import json
from mmcv import Config, DictAction
# config.merge_from_file(sys.argv[2])
# os.environ["CUDA_VISIBLE_DEVICES"] = \
#     ','.join((map(str, config.GPUS)))  # str(config.GPUS).strip('(').strip(')')

def parse_args():
    parser = argparse.ArgumentParser(description='Test crowd counting network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args
def patch_forward(model, img, dot_map, num_patches,mode):
    # crop the img and gt_map with a max stride on x and y axis
    # size: HW: __C_NWPU.TRAIN_SIZE
    # stack them with a the batchsize: __C_NWPU.TRAIN_BATCH_SIZE
    crop_imgs = []
    crop_dots, crop_masks = {},{}

    crop_dots['1'],crop_dots['2'],crop_dots['4'],crop_dots['8'] = [],[],[],[]
    crop_masks['1'],crop_masks['2'],crop_masks['4'],crop_masks['8'] = [], [], [],[]
    b, c, h, w = img.shape
    rh, rw = 768, 1024

    # support for multi-scale patch forward
    for i in range(0, h, rh):
        gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
        for j in range(0, w, rw):
            gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)

            crop_imgs.append(img[:, :, gis:gie, gjs:gje])
            for res_i in range(len(dot_map)):
                gis_,gie_ = gis//2**res_i, gie//2**res_i
                gjs_,gje_ = gjs//2**res_i, gje//2**res_i
                crop_dots[str(2**res_i)].append(dot_map[res_i][:, gis_:gie_, gjs_:gje_])
                mask = torch.zeros_like(dot_map[res_i]).cpu()
                mask[:, gis_:gie_, gjs_:gje_].fill_(1.0)
                crop_masks[str(2**res_i)].append(mask)

    crop_imgs = torch.cat(crop_imgs, dim=0)
    for k,v in crop_dots.items():
        crop_dots[k] =  torch.cat(v, dim=0)
    for k,v in crop_masks.items():
        crop_masks[k] =  torch.cat(v, dim=0)

    # forward may need repeatng
    crop_losses = []
    crop_preds = {}
    crop_labels = {}
    crop_labels['1'],crop_labels['2'],crop_labels['4'],crop_labels['8'] = [],[],[],[]
    crop_preds['1'],crop_preds['2'],crop_preds['4'],crop_preds['8'] = [], [], [],[]
    nz, bz = crop_imgs.size(0), num_patches
    keys_pre = None

    for i in range(0, nz, bz):
        gs, gt = i, min(nz, i + bz)
        result = model(crop_imgs[gs:gt], [crop_dots[k][gs:gt] for k in crop_dots.keys() ],
                       mode)
        crop_pred = result['pre_den']
        crop_label =  result['gt_den']

        keys_pre = result['pre_den'].keys()
        for k in keys_pre:
            crop_preds[k].append(crop_pred[k].cpu())
            crop_labels[k].append(crop_label[k].cpu())

        crop_losses.append(result['losses'].mean())

    for k in keys_pre:
        crop_preds[k] =  torch.cat(crop_preds[k], dim=0)
        crop_labels[k] =  torch.cat(crop_labels[k], dim=0)


    # splice them to the original size

    result = {'pre_den': {},'gt_den':{}}

    for res_i, k in enumerate(keys_pre):

        pred_map = torch.zeros_like(dot_map[res_i]).unsqueeze(0).cpu().float()
        labels = torch.zeros_like(dot_map[res_i]).unsqueeze(0).cpu().float()
        idx =0
        for i in range(0, h, rh):
            gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
            for j in range(0, w, rw):
                gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)

                gis_,gie_ = gis//2**res_i, gie//2**res_i
                gjs_,gje_ = gjs//2**res_i, gje//2**res_i

                pred_map[:,:, gis_:gie_, gjs_:gje_] += crop_preds[k][idx]
                labels[:,:, gis_:gie_, gjs_:gje_] += crop_labels[k][idx]
                idx += 1
        # import pdb
        # pdb.set_trace()
        # for the overlapping area, compute average value
        mask = crop_masks[k].sum(dim=0).unsqueeze(0).unsqueeze(0)
        pred_map = (pred_map / mask)
        labels = (labels / mask)
        result['pre_den'].update({k: pred_map} )
        result['gt_den'].update({k: labels} )
        result.update({'losses': crop_losses[0]} )
    return result
def test_cc(config, test_dataset, testloader, model
            ,mean, std, sv_dir='', sv_pred=False,logger=None):


    model.eval()
    save_count_txt = ''
    device = torch.cuda.current_device()
    scale_classes = 6
    cnt_errors = {'0': AverageMeter(),'1': AverageMeter(),'2': AverageMeter(),'3': AverageMeter(),'4': AverageMeter(),'5': AverageMeter() }
    with torch.no_grad():

        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, name = batch

            image, label, _, name = batch
            image = image.to(device)
            for i in range(len(label)):
                label[i] = label[i].to(device)

            # result = model(image, label, 'val')

            result = patch_forward(model, image, label,
                                   config.test.patch_batch_size, mode='val')

            with open(os.path.join(config.dataset.root, 'jsons', name[0]+'.json'), 'r') as f:
                info = json.load(f)
                boxes = info['boxes']

            losses=result['losses']
            pre_den=result['pre_den']['1']
            gt_den = result['gt_den']['1']

            for box in boxes:
                scale_mask = torch.zeros_like(pre_den)
                w_l, h_l, w_r, h_r = box
                area =  (w_r-w_l) * (h_r-h_l)
                hl,hr,wl,wr=int(h_l),int(h_r), int(w_l),int(w_r)
                pad_h=int(max(15-(hr-hl),0)//2)
                pad_w=int(max(15-(wr-wl),0)//2)
                scale_mask[:,:,hl-pad_h:hr+pad_h, wl-pad_w:wr+pad_w] =1
                gt_cnt = (gt_den*scale_mask).sum().item()
                pre_cnt = (pre_den*scale_mask).sum().item()
                instance_acc = 1-min(gt_cnt, abs(pre_cnt-gt_cnt))/(gt_cnt+1e-10)
                if area > 0 and area < 10**1:
                    cnt_errors['0'].update(instance_acc)

                if area > 10**1 and area < 10**2:
                    cnt_errors['1'].update(instance_acc)

                if area > 10**2 and area < 10**3:
                    cnt_errors['2'].update(instance_acc)

                if area > 10 **3and area < 10**4:
                    cnt_errors['3'].update(instance_acc)

                if area > 10**4 and area < 10**5:
                    cnt_errors['4'].update(instance_acc)

                if area > 10**5:
                    cnt_errors['5'].update(instance_acc)
            msg = ''
            for i in range(scale_classes):
                msg +='The mae of scale O:{}  smaple_num:{}\n'.format(cnt_errors[str(i)].avg,
                                                                      cnt_errors[str(i)].count)
            logger.info(msg)
            # import pdb
            # pdb.set_trace()

            #    -----------Counting performance------------------

    return  None, None, None

def main():

    args = parse_args()
    config = Config.fromfile(args.cfg)

    logger, final_output_dir = create_logger(
        config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    random_seed_setting(config)

    # build model
    device = torch.device('cuda:{}'.format(args.local_rank))

    model = Baseline_Counter(config.network,config.dataset.den_factor,config.train.route_size,device)

    # dump_input = torch.rand(
    #     (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    # )
    # logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

    if config.test.model_file:
        model_state_file = config.test.model_file
    else:
        model_state_file = os.path.join(final_output_dir,
                                        'final_state.pth')
    logger.info('=> loading model from {}'.format(model_state_file))

    pretrained_dict = torch.load(model_state_file)
    # model_dict = model.state_dict()

    # pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
    #                    if k[6:] in model_dict.keys()}
    # for k, _ in pretrained_dict.items():
    #     logger.info(
    #         '=> loading {} from pretrained model'.format(k))
    # model_dict.update(pretrained_dict)
    model.load_state_dict(pretrained_dict,strict=False)

    model = model.to(device)

    # prepare data
    test_dataset = eval('datasets.' + config.dataset.name)(
        root=config.dataset.root,
        list_path=config.dataset.test_set,
        num_samples=None,
        multi_scale=False,
        flip=False,
        base_size=config.test.base_size,
        downsample_rate=1)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,  # config.WORKERS,
        pin_memory=True)

    start = timeit.default_timer()
    if 'val' in config.dataset.test_set:

        mae, mse, nae,save_count_txt = test_cc(config, test_dataset, testloader, model,
                                               test_dataset.mean, test_dataset.std,
                                               sv_dir=final_output_dir, sv_pred=False

                                               ,logger=logger
                                               )

        msg = 'mae: {: 4.4f}, mse: {: 4.4f}, \
               nae: {: 4.4f}, Class IoU: '.format(mae,
                                                  mse, nae)
        logging.info(msg)


    with open(final_output_dir+'_submit_hloc.txt', 'w') as f:
        f.write(save_count_txt)

    end = timeit.default_timer()
    logger.info('Mins: %d' % np.int((end - start) / 60))
    logger.info('Done')


if __name__ == '__main__':
    main()
