
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------
import os
import pprint
import _init_paths
from lib.core.Counter import Counter
from lib.utils.utils import create_logger, random_seed_setting
from lib.utils.modelsummary import get_model_summary
from lib.core.cc_function import test_cc
import datasets
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


def parse_args():
    parser = argparse.ArgumentParser(description='Test crowd counting network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--checkpoint',
                    help='experiment configure file name',
                    required=True,
                    type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi','torchrun'],
                        default='none',
                        help='job launcher')
        
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

    logger.info('GPU idx:'+os.environ['CUDA_VISIBLE_DEVICES'])
    gpus = config.gpus
    distributed = torch.cuda.device_count() > 1
    if distributed:
        torch.cuda.set_device(args.local_rank)
        # torch.distributed.init_process_group(
        #     backend="nccl", init_method="env://",
        # )
        init_dist(args.launcher)
        # torch.cuda.set_device(args.local_rank)
        if args.launcher == 'pytorch':
            args.local_rank = int(os.environ["LOCAL_RANK"])
        else:
            rank, world_size = get_dist_info()
            args.local_rank = rank
    
    # cudnn related setting
    random_seed_setting(config)

    # build model
    device = torch.device('cuda:{}'.format(args.local_rank))

    model = Baseline_Counter(config.network,config.dataset.den_factor,config.train.route_size,device)

    # dump_input = torch.rand(
    #     (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    # )
    # logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

    if args.checkpoint:
        model_state_file = args.checkpoint
    elif config.test.model_file:
        model_state_file = config.test.model_file
    else:
        model_state_file = os.path.join(final_output_dir,
                                        'final_state.pth')
    
    logger.info('=> loading model from {}'.format(model_state_file))

    pretrained_dict = torch.load(model_state_file)


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
    if 'test' in  config.dataset.test_set or 'val' in config.dataset.test_set:

        mae, mse, nae,save_count_txt = test_cc(config, test_dataset, testloader, model,
                                test_dataset.mean, test_dataset.std,
                                sv_dir=final_output_dir, sv_pred=False,logger=logger
                                )

        msg = 'mae: {: 4.4f}, mse: {: 4.4f}, \
               nae: {: 4.4f}, Class IoU: '.format(mae,
                                                  mse, nae)
        logging.info(msg)


    with open(final_output_dir+'_submit_SIC.txt', 'w') as f:
        f.write(save_count_txt)

    end = timeit.default_timer()
    logger.info('Mins: %d' % np.int32((end - start) / 60))
    logger.info('Done')


if __name__ == '__main__':
    main()
