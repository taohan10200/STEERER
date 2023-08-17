import torch
import _init_paths
from lib.config import config
from lib.config import update_config
import argparse
from lib.utils.modelsummary import get_model_summary

from lib.models.seg_hrnet_sum import  get_seg_model
from lib.models.vit import  get_seg_model
def parse_args():
    parser = argparse.ArgumentParser(description='Train Crowd Counting network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

# args = parse_args()
#
# model = get_seg_model(config)
# # dump_input = torch.rand(
# #     (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
# # )
# dump_input = torch.rand(
#     (1, 3, 224, 224)
# )
# print(get_model_summary(model.to(0), dump_input.to(0)))
import os
from lib.models.backbones import HRTBackbone_cc, CCFBackbone, \
    MocBackbone, HRNetBackbone, MocHRBackbone, HRTBackbone,MocCatBackbone,\
    MAEvitBackbone, VGGBackbone

from lib.utils.modelsummary import get_model_summary
# from lib.utils.flop_count import flop_count
from fvcore.nn.flop_count import flop_count
from fvcore.nn.parameter_count import parameter_count_table
from mmcv import  Config
dump_input = torch.rand(
    (1, 3, 768, 768)
)
os.environ["drop_path_rate"] = '0'
configer = Config.fromfile('./configs/count/VGG/SHHA_vgg19_bn.py')
model   = VGGBackbone(configer.network)()

print(parameter_count_table(model))
print(flop_count(model.cuda(), (dump_input.cuda(),)))
print(get_model_summary(model.cuda(), dump_input.cuda()))

import pdb
pdb.set_trace()

model   = MAEvitBackbone(configer.network)()

print(parameter_count_table(model))
print(flop_count(model.cuda(), (dump_input.cuda(),)))
print(get_model_summary(model.cuda(), dump_input.cuda()))

import pdb
pdb.set_trace()

# print(configer)
model   = HRTBackbone(configer.network)()

print(parameter_count_table(model))
print(flop_count(model.cuda(), (dump_input.cuda(),)))
print(get_model_summary(model.cuda(), dump_input.cuda()))

import pdb
pdb.set_trace()




#
#
# configer = Config.fromfile('./configs/imagenet/ccformer_b.py')
#
# model   = CCFBackbone(configer.network)()
# print(get_model_summary(model.cuda(), dump_input.cuda()))
# model_duplicate = model

#




configer = Config.fromfile('./configs/count/SHHB_mocHR_small.py')

model   = MocHRBackbone(configer.network)()
print(parameter_count_table(model))
print(get_model_summary(model.cuda(), dump_input.cuda()))

print(flop_count(model.cuda(), (dump_input.cuda(),)))

configer = Config.fromfile('./configs/count/SHHB_mocHR_small.py')

model   = MocCatBackbone(configer.network)()
print(parameter_count_table(model))
print(get_model_summary(model.cuda(), dump_input.cuda()))

print(flop_count(model.cuda(), (dump_input.cuda(),)))


# configer = Config.fromfile('./configs/imagenet/hrnet.py')

# model   = HRNetBackbone(configer.network)()
# print(get_model_summary(model.cuda(), dump_input.cuda()))
#
# print(flop_count(model.cuda(), (dump_input.cuda(),)))


# configer = Config.fromfile('./configs/imagenet/moc_base.py')
#
# model   = MocBackbone(configer.network)()
# print(get_model_summary(model.cuda(), dump_input.cuda()))


# model;
import pdb
pdb.set_trace()