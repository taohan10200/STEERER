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

def target_transform(gt_point, rate):
    point_map = gt_point.cpu().numpy()
    pts = np.array(list(zip(np.nonzero(point_map)[2], np.nonzero(point_map)[1])))
    pt2d = np.zeros((int(rate * point_map.shape[1]) + 1, int(rate * point_map.shape[2]) + 1), dtype=np.float32)

    for i, pt in enumerate(pts):
        pt2d[int(rate * pt[1]), int(rate * pt[0])] = 1.0

    return pt2d


def gt_transform(pt2d, cropsize, rate):
    from scipy.ndimage.filters import gaussian_filter
    import scipy
    [x, y, w, h] = cropsize
    pt2d = pt2d[int(y * rate):int(rate * (y + h)), int(x * rate):int(rate * (x + w))]
    density = np.zeros((int(pt2d.shape[0]), int(pt2d.shape[1])), dtype=np.float32)
    pts = np.array(list(zip(np.nonzero(pt2d)[1], np.nonzero(pt2d)[0])))
    orig = np.zeros((int(pt2d.shape[0]), int(pt2d.shape[1])), dtype=np.float32)
    for i, pt in enumerate(pts):
        orig[int(pt[1]), int(pt[0])] = 1.0

    density += scipy.ndimage.filters.gaussian_filter(orig, 4, mode='constant')
    # print(np.sum(density))
    return density
def test_cc(config, test_dataset, testloader, model,rate_model,gaussian
            ,mean, std, sv_dir='', sv_pred=False,logger=None):
    from scale_generalization.autoscale.find_couter import findmaxcontours

    model.eval()
    save_count_txt = ''
    device = torch.cuda.current_device()
    scale_classes = 6
    cnt_errors = {'0': AverageMeter(),'1': AverageMeter(),'2': AverageMeter(),'3': AverageMeter(),'4': AverageMeter(),'5': AverageMeter() }
    with torch.no_grad():

        for index, batch in enumerate(tqdm(testloader)):
            img, label, scale_factor, fname = batch

            img = img.to(device)
            for i in range(len(label)):
                label[i] = label[i].to(device)
            # kpoint


            # result = model(image, label, 'val')

            # for i, (img,target,kpoint,fname,sigma_map) in enumerate(test_loader):

            target = label[0]
            target = gaussian(target.unsqueeze(1))
            d2, d3, d4, d5, d6, fs = model(img, target.squeeze(1), refine_flag=True)
            target = target.unsqueeze(1)


            d6 = F.relu(d6)
            density_map = d6.data.cpu().numpy()

            original_count = density_map.sum()
            original_density = d6


            pred_color_map= original_density.data.cpu().numpy()
            pred_color_map = cv2.applyColorMap(
            (255 * pred_color_map / (pred_color_map.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
            SAV_DIR = '/mnt/petrelfs/hantao/2023/Autoscale/AutoScale_regression-f0c1583123c16bb726b239c5625834e7b01b17d0/output_NWPU'
            os.makedirs(SAV_DIR,exist_ok=True)
            cv2.imwrite('{}/{}.png'.format( SAV_DIR,fname), pred_color_map)


            [x, y, w, h] = findmaxcontours(density_map, fname)

            rate_feature = F.adaptive_avg_pool2d(fs[:, :, y:(y + h), x:(x + w)], (14, 14))
            rate = rate_model(rate_feature).clamp_(0.5, 9)
            rate = torch.sqrt(rate)

            if (float(w * h) / (img.size(2) * img.size(3))) > 0.1:

                img_pros = img[:, :, y:(y + h), x:(x + w)]

                img_transed = F.upsample_bilinear(img_pros, scale_factor=rate.item())

                # import pdb
                # pdb.set_trace()

                pt2d = target_transform(label[0], rate)
                target_choose = gt_transform(pt2d, [x, y, w, h], rate.item())

                target_choose = torch.from_numpy(target_choose).type(torch.FloatTensor).unsqueeze(0)

                dd2, dd3, dd4, dd5, dd6 = model(img_transed, target_choose, refine_flag=False)

                # dd6[dd6<0]=0
                temp = dd6.data.cpu().numpy().sum()
                original_density[:, :, y:(y + h), x:(x + w)] = 0
                count = original_density.data.cpu().numpy().sum() + temp
            else:
                count = d6.data.cpu().numpy().sum()

            with open(os.path.join(config.dataset.root, 'jsons', fname[0]+'.json'), 'r') as f:
                info = json.load(f)
                boxes = info['boxes']


            pre_den= d6
            gt_den =target

            for box in boxes:
                scale_mask = torch.zeros_like(pre_den)
                w_l, h_l, w_r, h_r = box
                area =  (w_r-w_l) * (h_r-h_l)
                scale_mask[:,:,int(h_l*scale_factor):int(h_r*scale_factor), int(w_l*scale_factor):int(w_r*scale_factor)] =1

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
    from scale_generalization.autoscale.fpn import AutoScale
    from scale_generalization.autoscale.rate_model import RATEnet
    from lib.utils.Gaussianlayer import Gaussianlayer
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

    model = AutoScale().to(device)
    model = nn.DataParallel(model)

    rate_model = RATEnet().to(device)
    rate_model = nn.DataParallel(rate_model)


    gaussian = Gaussianlayer([4], 15).to(device)

    if config.test.model_file:
        if os.path.isfile(config.test.model_file):
            print("=> loading checkpoint '{}'".format(config.test.model_file))

            path= config.test.model_file
            # checkpoint = torch.load(args.pre, map_location=lambda storage, loc: storage, pickle_module=pickle)
            checkpoint = torch.load(path)
            # import pdb
            # pdb.set_trace()
            model.load_state_dict(checkpoint['state_dict'])
            rate_model.load_state_dict(checkpoint['rate_state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))

    # model_dict = model.state_dict()

    # pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
    #                    if k[6:] in model_dict.keys()}
    # for k, _ in pretrained_dict.items():
    #     logger.info(
    #         '=> loading {} from pretrained model'.format(k))
    # model_dict.update(pretrained_dict)

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
    if 'val' or 'test' in config.dataset.test_set:

        mae, mse, nae,save_count_txt = test_cc(config, test_dataset, testloader, model,rate_model,gaussian,
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
