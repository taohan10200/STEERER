# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Rainbowsecret (yuyua@microsoft.com)
# "High-Resolution Representations for Labeling Pixels and Regions"
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_, DropPath

import logging
Log = logging.getLogger(__name__)

if torch.__version__.startswith("1"):
    relu_inplace = True
else:
    relu_inplace = False
from lib.models.backbones.modules.cnn_blocks import HrnetBasicBlock
from lib.models.backbones.modules.cnn_blocks import ConvxBasicBlock
from lib.models.backbones.modules.cnn_blocks import Bottleneck, LayerNorm
from lib.models.backbones.modules.transformer_block import GeneralTransformerBlock


blocks_dict = {"HRNetBASIC": HrnetBasicBlock,
               "BOTTLENECK": Bottleneck,
               "TRANSFORMER_BLOCK":GeneralTransformerBlock}
norm_dict = {"BN":nn.BatchNorm2d , "LN":LayerNorm}
activation_dict = {"ReLU":nn.ReLU(inplace=True) , "LN":LayerNorm}

class HighResolutionModule(nn.Module):
    def __init__(
        self,
        layer_cfg,
        num_inchannels,
        norm,
        activation,
        multi_scale_output=True,
        dp_rates_4module: list= [0.0],
        module_idx=0,
    ):
        super(HighResolutionModule, self).__init__()

        self.layer_cfg = layer_cfg
        self.num_branches = layer_cfg["NUM_BRANCHES"]
        self.num_blocks = layer_cfg["NUM_BLOCKS"]
        self.num_channels = layer_cfg["NUM_CHANNELS"]
        self.block = blocks_dict[layer_cfg["BLOCK"]]
        self.fuse_method = layer_cfg["FUSE_METHOD"]
        self.expansion = layer_cfg["EXPANSION"]

        self.num_inchannels = num_inchannels

        self.Norm = norm
        self.Activation = activation
        self.module_idx = module_idx
        self.multi_scale_output = multi_scale_output
        self._check_branches(
            self.num_branches, self.num_blocks, self.num_inchannels, self.num_channels
        )

        self.branches = self._make_branches(
            dp_rates_4module,
        )
        if multi_scale_output:
            # self.fuse_layers = self._make_fuse_layers(self.module_idx)
            self.fuse_layers = None
        else:
            self.fuse_layers = None
    def _check_branches(
        self, num_branches, num_blocks, num_inchannels, num_channels
    ):
        if num_branches != len(num_blocks):
            error_msg = "NUM_BRANCHES({}) <> NUM_BLOCKS({})".format(
                num_branches, len(num_blocks)
            )
            Log.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = "NUM_BRANCHES({}) <> NUM_CHANNELS({})".format(
                num_branches, len(num_channels)
            )
            Log.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = "NUM_BRANCHES({}) <> NUM_INCHANNELS({})".format(
                num_branches, len(num_inchannels)
            )
            Log.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(
        self,
        branch_index,
        dp_rates_4module: list= [0.0],
    ):
        layers = []
        if self.block is GeneralTransformerBlock:
            layers.append(
                self.block(
                    self.num_inchannels[branch_index],
                    self.num_inchannels[branch_index],
                    num_heads=self.layer_cfg["NUM_HEADS"][branch_index],
                    window_size=self.layer_cfg["NUM_WINDOW_SIZES"][branch_index],
                    mlp_ratio=self.layer_cfg["NUM_MLP_RATIOS"][branch_index],
                    drop_path=dp_rates_4module[0],
                )
            )
        else:
            layers.append(
                self.block(
                    self.num_inchannels[branch_index],
                    self.num_inchannels[branch_index],
                    drop_path=dp_rates_4module[0],
                    expansion=self.expansion[branch_index]
                )
            )

        for i in range(1, self.num_blocks[branch_index]):
            if self.block is GeneralTransformerBlock:
                layers.append(
                    self.block(
                        self.num_inchannels[branch_index],
                        self.num_inchannels[branch_index],
                        num_heads=self.layer_cfg["NUM_HEADS"][branch_index],
                        window_size=self.layer_cfg["NUM_WINDOW_SIZES"][branch_index],
                        mlp_ratio=self.layer_cfg["NUM_MLP_RATIOS"][branch_index],
                        drop_path=dp_rates_4module[i],
                    )
            )
            else:
                layers.append(
                    self.block(
                        self.num_inchannels[branch_index],
                        self.num_inchannels[branch_index],
                        drop_path=dp_rates_4module[i],
                        expansion=self.expansion[branch_index]
                    )
                )

        return nn.Sequential(*layers)

    def _make_branches(
        self, dp_rates_4module: list= [0.0],
    ):
        branches = []

        for i in range(self.num_branches):
            branches.append(
                self._make_one_branch(
                    i,
                    dp_rates_4module,
                )
            )
        return nn.ModuleList(branches)

    def _make_fuse_layers(self, module_idx):
        if self.num_branches == 1 :
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        # import pdb
        # pdb.set_trace()
        for i in range(num_branches):
            fuse_layer = []
            if module_idx<3  and i ==  module_idx + 1:
                for j in range(num_branches):
                    if j==i-1: #else
                        fuse_layer.append(
                            nn.Sequential(nn.Sequential(
                                nn.Conv2d(
                                    num_inchannels[j],
                                    num_inchannels[i],
                                    3,
                                    2,
                                    1
                                ),
                                self.Norm(num_inchannels[i]),
                                self.Activation,)
                                        ))
                    elif j == i:
                        fuse_layer.append(nn.Identity())
                fuse_layers.append(nn.ModuleList(fuse_layer))
            elif module_idx>=3  and num_branches-1-i ==  module_idx-3+1:
                for j in range(num_branches):
                    if j == i:
                        fuse_layer.append(nn.Identity())
                    if j ==i+1: #if j > i:
                        fuse_layer.append(
                            nn.Sequential(
                                nn.Conv2d(
                                    num_inchannels[j],
                                    num_inchannels[i],
                                    1,
                                    1,
                                    0
                                ),
                                self.Norm(num_inchannels[i]),
                                nn.Upsample(scale_factor=2 ** (j - i), mode="bilinear"),
                                self.Activation,
                            )
                        )

                fuse_layers.append(nn.ModuleList(fuse_layer))
            else:
                fuse_layers.append(nn.ModuleList([nn.Identity()]))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        # import pdb
        # pdb.set_trace()
        for i in range(self.num_branches):
            if x[i] is not None:
                x[i] = self.branches[i](x[i])
        # import pdb
        # pdb.set_trace()
        if self.fuse_layers is  None:
            return  x
        else:
            for i in range(len(self.fuse_layers)):
                last_layer_input = x[max(0, i-1)]
                cur_layer_input = x[i]
                next_layer_input = x[min(len(self.fuse_layers)-1, i+1)]

                if len(self.fuse_layers[i])==1:
                    if  cur_layer_input is not None:
                        x[i]=self.fuse_layers[i][0](x[i])
                    else:
                        x[i]=None
                else:
                    # import pdb
                    # pdb.set_trace()
                    if  not isinstance(self.fuse_layers[i][0],nn.Identity):

                        if  last_layer_input is None and cur_layer_input is None:
                            x[i]=None

                        elif last_layer_input is None and cur_layer_input is not None:
                            x[i] = self.fuse_layers[i][1](x[i])

                        elif last_layer_input is not None and cur_layer_input is None:
                            x[i] = self.fuse_layers[i][0](x[i-1])

                    else:
                        if next_layer_input is not None and cur_layer_input is not None:
                            y = self.fuse_layers[i][0](x[i])
                            y =y + self.fuse_layers[i][1](x[i+1])
                            x[i] = y
                            x[i+1]=None

            return x



class HighResolutionNet(nn.Module):
    def __init__(self, cfg, bn_type, bn_momentum, **kwargs):
        self.inplanes = 64
        self.cfg =cfg
        self.drop_path_rate = cfg['DROP_PATH_RATE']
        super(HighResolutionNet, self).__init__()
        Norm= norm_dict[cfg.NORM]
        Activation = activation_dict[cfg.ACTIVATION]

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,  bias=False),
            Norm(64),
            Activation,
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,bias=False),
            Norm(64),
            Activation
        )

        self.layer1 = self._make_layer(
            blocks_dict[cfg.STEM_BLOCK], dim=64,blocks=2, drop_path=0, channel_expansion=cfg["STEM_EXPANSION"])


        self.stage2_cfg = cfg["STAGE2"]

        depths = sum([self.cfg[stage]["NUM_BLOCKS"][0]*self.cfg[stage]["NUM_MODULES"]
                      for stage in ["STAGE2","STAGE3","STAGE4"]])

        dp_rates=[x.item() for x in torch.linspace(0, self.drop_path_rate, depths)]

        # block = blocks_dict[self.stage2_cfg["BLOCK"]]

        self.transition1 = self._make_transition_layer(
            [64], cfg["STAGE2"]["NUM_CHANNELS"], Norm=Norm, Activation=Activation
        )

        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, Norm, Activation, dp_rates=dp_rates)


        self.transition2 = self._make_transition_layer(
            pre_stage_channels, cfg["STAGE3"]['NUM_CHANNELS'], Norm=Norm, Activation=Activation
        )
        self.stage3, pre_stage_channels = self._make_stage(
            cfg["STAGE3"], Norm, Activation, dp_rates=dp_rates)


        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, Norm, Activation, dp_rates=dp_rates)

        self.transition2 = self._make_transition_layer(
            pre_stage_channels, cfg["STAGE3"]['NUM_CHANNELS'], Norm=Norm, Activation=Activation
        )
        self.stage3, pre_stage_channels = self._make_stage(
            cfg["STAGE3"], Norm, Activation, dp_rates=dp_rates)

        self.transition3 = self._make_transition_layer(
            pre_stage_channels, cfg["STAGE4"]['NUM_CHANNELS'], Norm=Norm, Activation=Activation
        )
        self.stage4, pre_stage_channels = self._make_stage(
            cfg["STAGE4"], Norm, Activation, dp_rates=dp_rates)


        if os.environ.get("keep_imagenet_head"):
            (
                self.incre_modules,
                self.downsamp_modules,
                self.final_layer,
            ) = self._make_head(
                pre_stage_channels, bn_type=bn_type, bn_momentum=bn_momentum
            )


    def _make_transition_layer(
        self, num_channels_pre_layer, num_channels_cur_layer, Norm,Activation
    ):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3,
                                1,
                                1,
                                bias=False
                            ),
                            Norm(num_channels_cur_layer[i]),
                            Activation
                            ),
                        )
            else:
                conv3x3s = []
                # kernel_size = 2**(i-num_branches_pre+1)
                inchannels = num_channels_pre_layer[num_branches_pre-1]
                outchannels = num_channels_cur_layer[num_branches_pre]
                for j in range(i-num_branches_pre+1):
                    conv3x3s.append(
                                nn.Sequential
                                (nn.Conv2d(inchannels, outchannels, 3, stride=2, padding=1,  bias=False),
                                Norm(outchannels),
                                Activation)
                            )
                    inchannels = num_channels_cur_layer[num_branches_pre+j]
                    outchannels = num_channels_cur_layer[min(num_branches_pre+j+1, num_branches_cur-1)]
                transition_layers.append(
                    nn.Sequential(*conv3x3s)
                )

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, blocks, dim, drop_path=0, channel_expansion=4):
        downsample = None

        layers = []
        layers.append(
            block(dim, dim, drop_path, expansion=channel_expansion)
        )
        for i in range(1, blocks):
            layers.append(
                block(dim, dim, drop_path=0, expansion=channel_expansion)
            )
        # layers.append(LayerNorm(dim, eps=1e-6, data_format="channels_first"))

        return nn.Sequential(*layers)

    def _make_stage(
        self,
        layer_config,
        norm,
        activation,
        dp_rates: list = [0.],
    ):
        num_modules = layer_config["NUM_MODULES"]
        num_inchannels =  layer_config["NUM_CHANNELS"]
        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if  i == num_modules - 1:
                multi_scale_output = False
            else:
                multi_scale_output = True

            dp_rates_4module = dp_rates[sum([layer_config["NUM_BLOCKS"][0] for j in range(i)]):]

            modules.append(
                HighResolutionModule(
                    layer_config,
                    num_inchannels,
                    norm,
                    activation,
                    multi_scale_output=multi_scale_output,
                    dp_rates_4module=dp_rates_4module,
                    module_idx=i,
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x, which_branch=None):

        x = self.stem(x)
        x = self.layer1(x)



        x_list = []
        if which_branch is None:

            out_list = [None, None, None, None]
            branch_in0 = [None, None, None, None]
            branch_in1 = [None, None, None, None]
            branch_in2 = [None, None, None, None]
            branch_in3 = [None, None, None, None]
            for i in range(4):
                if i == 0:
                    branch_in1[i+1] = self.transition1[i+1](x)

                    branch_in0[i] = self.transition1[i](x)
                    branch_in0[i] = self.stage2(branch_in0)[i]

                    branch_in0[i] = self.transition2[i](branch_in0[i])
                    branch_in0[i] =(self.stage3(branch_in0)[i])

                    branch_in0[i] = self.transition3[i](branch_in0[i])
                    x_list.append(self.stage4(branch_in0)[i])

                if i == 1:

                    branch_in1[i] = self.stage2(branch_in1)[i]
                    branch_in2[i+1] = self.transition2[i+1](branch_in1[i])

                    branch_in1[i] = self.transition2[i](branch_in1[i])
                    branch_in1[i] = self.stage3(branch_in1)[i]

                    branch_in1[i] = self.transition3[i](branch_in1[i])
                    x_list.append(self.stage4(branch_in1)[i])

                if i == 2:
                    branch_in2[i] = self.stage3(branch_in2)[i]
                    branch_in3[i+1] = self.transition3[i+1](branch_in2[i])

                    branch_in2[i] = self.transition3[i](branch_in2[i])
                    x_list.append(self.stage4(branch_in2)[i])

                if i == 3:

                    x_list.append(self.stage4(branch_in3)[i])
                #
                # import pdb
                # pdb.set_trace()

            # x_list = []
            # for i in range(self.cfg["STAGE3"]["NUM_BRANCHES"]):
            #     in_list = [None, None, None, None]
            #     in_list[i] = self.transition2[i](x_list[i])
            #     x_list.append(self.stage2(in_list)[i])

            # import pdb
            # pdb.set_trace()

            return  x_list




        in_list = [None, None, None, None]
        in_list[which_branch] = self.transition1[which_branch](x)
        x_list = self.stage2(in_list)[which_branch:]
        return x_list

    def get_stem_feature(self,x):
        x = self.stem(x)
        x = self.layer1(x)
        return x

    def moe_singlePath(self, stem, which_branch=None):
        # in_list = [None, None, None, None]
        # in_list[which_branch] = self.transition1[which_branch](stem)
        # x_list = self.stage2(in_list)[:which_branch+1]

        branch_in0 = [None, None, None, None]
        branch_in1 = [None, None, None, None]
        branch_in2 = [None, None, None, None]
        branch_in3 = [None, None, None, None]
        if which_branch == 0:
            branch_in0[0] = self.transition1[0](stem)
            branch_in0[0] = self.stage2(branch_in0)[0]

            branch_in0[0] = self.transition2[0](branch_in0[0])
            branch_in0[0] =(self.stage3(branch_in0)[0])

            branch_in0[0] = self.transition3[0](branch_in0[0])
            x_list=self.stage4(branch_in0)

        if which_branch == 1:
            branch_in1[1] = self.transition1[1](stem)
            branch_in1[1] = self.stage2(branch_in1)[1]

            branch_in1[1] = self.transition2[1](branch_in1[1])
            branch_in1[1] = self.stage3(branch_in1)[1]

            branch_in1[1] = self.transition3[1](branch_in1[1])
            x_list=self.stage4(branch_in1)


        if which_branch == 2:
            branch_in1[1] = self.transition1[1](stem)
            branch_in1[1] = self.stage2(branch_in1)[1]
            branch_in2[2] = self.transition2[2](branch_in1[1])

            branch_in2[2] = self.stage3(branch_in2)[2]

            branch_in2[2] = self.transition3[2](branch_in2[2])
            x_list = self.stage4(branch_in2)

        if which_branch == 3:
            branch_in1[1] = self.transition1[1](stem)
            branch_in1[1] = self.stage2(branch_in1)[1]
            branch_in2[2] = self.transition2[2](branch_in1[1])

            branch_in2[2] = self.stage3(branch_in2)[2]

            branch_in3[3] = self.transition3[3](branch_in2[2])
            x_list=self.stage4(branch_in3)
        else:
            assert ValueError('which_branch must be in one of the following [1,2,3,4]')

        return  x_list

    def init_weights(self, pretrained='',):
        Log.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, (nn.Conv2d,nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)

            Log.info('=> loading pretrained model {}'.format(pretrained))

            model_dict = self.state_dict()

            load_dict = {k[9:]: v for k, v in pretrained_dict.items()
                         if k[9:] in model_dict.keys()}

            # for k, _ in load_dict.items():
            #     logger.info(
            #         '=> loading {} pretrained model {}'.format(k, pretrained))

            Log.info(
                "Missing keys: {}".format(list(set(model_dict) - set(load_dict)
                                               )))
            #

            model_dict.update(load_dict)
            self.load_state_dict(model_dict)




class MocBackbone(object):
    def __init__(self, configer):
        self.configer = configer

    def __call__(self):
        arch = self.configer.sub_arch
        from lib.models.backbones.hrnet.moc_config import MODEL_CONFIGS

        if arch in [
            "moc_small",
            "moc_base",
            "moct_small",
        ]:
            arch_net = HighResolutionNet(
                MODEL_CONFIGS[arch], bn_type="torchbn", bn_momentum=0.1
            )

            arch_net.init_weights(pretrained=self.configer.pretrained_backbone)

        else:
            raise Exception("Architecture undefined!")

        return arch_net


