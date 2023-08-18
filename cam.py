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
from lib.core.cc_function import testval, test
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

from mmcv import Config, DictAction
from torchcam.methods import SmoothGradCAMp
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
    model_dict = model.state_dict()
    # pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
    #                    if k[6:] in model_dict.keys()}
    # for k, _ in pretrained_dict.items():
    #     logger.info(
    #         '=> loading {} from pretrained model'.format(k))
    # model_dict.update(pretrained_dict)
    model.load_state_dict(pretrained_dict)

    model = model.to(device)

    # prepare data
    test_dataset = eval('datasets.' + config.dataset.name)(
        root=config.dataset.root,
        list_path=config.dataset.test_set,
        num_samples=None,
        multi_scale=False,
        flip=False,
        base_size=config.test.loc_base_size,
        downsample_rate=1)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # config.WORKERS,
        pin_memory=True)

    def read_box_gt(box_gt_file):
        gt_data = {}
        with open(box_gt_file) as f:
            for line in f.readlines():
                line = line.strip().split(' ')

                line_data = [int(i) for i in line]
                idx, num = [line_data[0], line_data[1]]
                points_r = []
                if num > 0:
                    points_r = np.array(line_data[2:]).reshape(((len(line) - 2) // 5, 5))
                    gt_data[idx] = {'num': num, 'points': points_r[:, 0:2], 'sigma': points_r[:, 2:4], 'level': points_r[:, 4]}
                else:
                    gt_data[idx] = {'num': 0, 'points': [], 'sigma': [], 'level': []}

        return gt_data
    loc_gt = read_box_gt(os.path.join(config.dataset.root, config.dataset.loc_gt))

    start = timeit.default_timer()


    model.eval()
    device = torch.cuda.current_device()
    cam_extractor = SmoothGradCAMpp(model)
    import pdb
    pdb.set_trace()
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, name = batch

            image, label, _, name = batch
            image = image.to(device)
            for i in range(len(label)):
                label[i] = label[i].to(device)

            # pad_dims = (0, pad_w, 0, pad_h)
            # image = F.pad(image, pad_dims, "constant")
            # label = F.pad(label, pad_dims, "constant")
            result = model(image, label, 'val')

            losses=result['losses']
            pre_den=result['pre_den_x1']
            gt_den = result['gt_den_x1']
            # losses, pred, labels = patch_forward(model, image, label,
            #                                       config.TEST.PATCH_BATCH_SIZE, mode='val')
            # import pdb
            # pdb.set_trace()


            #    -----------Counting performance------------------
            gt_count, pred_cnt = label[0].sum().item(), pre_data['num'] #pred.sum().item()
            msg = '{} {}' .format(gt_count,pred_cnt)
            logger.info(msg)
            # print(name,':', gt_count, pred_cnt)
            s_mae = abs(gt_count - pred_cnt)
            s_mse = ((gt_count - pred_cnt) * (gt_count - pred_cnt))



            image = image[0]

            for t, m, s in zip(image, mean, std):
                t.mul_(s).add_(m)
            save_results_more(name, sv_dir, image.cpu().data, \
                              pre_den[0].detach().cpu(), gt_den[0].detach().cpu(),pred_cnt,gt_count,
                              pre_data['points'])


        msg = 'mae: {: 4.4f}, mse: {: 4.4f}, \
               nae: {: 4.4f}, Class IoU: '.format(mae,
                                                  mse, nae)
        logging.info(msg)
        # logging.info(IoU_array)


    end = timeit.default_timer()
    logger.info('Mins: %d' % np.int((end - start) / 60))
    logger.info('Done')


if __name__ == '__main__':
    main()
