# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn.functional as F

from lib_cls.models.losses import Accuracy
from ..builder import HEADS, build_loss
from ..utils import is_tracing
from .base_head import BaseHead
from lib.utils.logger import Logger as Log

@HEADS.register_module()
class ClsHead(BaseHead):
    """classification head.

    Args:
        loss (dict): Config of classification loss.
        topk (int | tuple): Top-k accuracy.
        cal_acc (bool): Whether to calculate accuracy during training.
            If you use Mixup/CutMix or something like that during training,
            it is not reasonable to calculate accuracy. Defaults to False.
    """

    def __init__(self,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk=(1, ),
                 cal_acc=False,
                 init_cfg=None):
        super(ClsHead, self).__init__(init_cfg=init_cfg)

        assert isinstance(loss, dict)
        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk, )
        for _topk in topk:
            assert _topk > 0, 'Top-k should be larger than 0'
        self.topk = topk

        self.compute_loss = build_loss(loss)
        self.compute_accuracy = Accuracy(topk=self.topk)
        self.cal_acc = cal_acc

    def loss(self, cls_score, gt_label, **kwargs):
        num_samples = len(cls_score)
        losses = dict()
        # compute loss
        loss = self.compute_loss(
            cls_score, gt_label, avg_factor=num_samples, **kwargs)
        if self.cal_acc:
            # compute accuracy
            acc = self.compute_accuracy(cls_score, gt_label)
            assert len(acc) == len(self.topk)
            losses['accuracy'] = {
                f'top-{k}': a
                for k, a in zip(self.topk, acc)
            }
        losses['loss'] = loss
        return losses

    def forward_train(self, cls_score, gt_label, **kwargs):
        if isinstance(cls_score, tuple):
            cls_score = cls_score[-1]
        losses = self.loss(cls_score, gt_label, **kwargs)
        return losses

    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]

        warnings.warn(
            'The input of ClsHead should be already logits. '
            'Please modify the backbone if you want to get pre-logits feature.'
        )
        return x

    def simple_test(self, cls_score, softmax=True, post_process=True):
        """Inference without augmentation.

        Args:
            cls_score (tuple[Tensor]): The input classification score logits.
                Multi-stage inputs are acceptable but only the last stage will
                be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            softmax (bool): Whether to softmax the classification score.
            post_process (bool): Whether to do post processing the
                inference results. It will convert the output to a list.

        Returns:
            Tensor | list: The inference results.

                - If no post processing, the output is a tensor with shape
                  ``(num_samples, num_classes)``.
                - If post processing, the output is a multi-dimentional list of
                  float and the dimensions are ``(num_samples, num_classes)``.
        """
        if isinstance(cls_score, tuple):
            cls_score = cls_score[-1]

        if softmax:
            pred = (
                F.softmax(cls_score, dim=1) if cls_score is not None else None)
        else:
            pred = cls_score

        if post_process:
            return self.post_process(pred)
        else:
            return pred

    def post_process(self, pred):
        on_trace = is_tracing()
        if torch.onnx.is_in_onnx_export() or on_trace:
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred

import torch.nn as nn
from lib.models.backbones.modules.cnn_blocks import Bottleneck, BottleneckDWP
from lib.models.backbones.modules.transformer_block import GeneralTransformerBlock
BN_MOMENTUM = 0.1

blocks_dict = {
    "BOTTLENECK": Bottleneck,
    "TRANSFORMER_BLOCK": GeneralTransformerBlock,
}


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
class MocClsHead(nn.Module):
    def __init__(self, configer):
        super(MocClsHead, self).__init__()
        self.pre_stage_channels = configer.in_channels
        self.num_classes = configer.num_classes
        self.downsamp_modules, self.final_layer = self._make_head(
            self.pre_stage_channels
        )
        self.classifier = nn.Linear(2048, self.num_classes)
    def _make_head(self, pre_stage_channels):
        Log.info("pre_stage_channels: {}".format(pre_stage_channels))
        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels =  self.pre_stage_channels[i]
            out_channels =  self.pre_stage_channels[i + 1]
            downsamp_module = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels= self.pre_stage_channels[-1],
                out_channels=2048,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(2048),
            nn.ReLU()
        )

        return downsamp_modules, final_layer

    def forward(self,y_list):

        y = y_list[0]

        for i in range(len(self.downsamp_modules)):
            if y_list[i+1] is not None:
                y = y_list[i + 1] + self.downsamp_modules[i](y)
            else:
                y= self.downsamp_modules[i](y)
        y = self.final_layer(y)
        y = F.avg_pool2d(y, kernel_size=y.size()[2:]).view(y.size(0), -1)
        y = self.classifier(y)
        return y


class HrtClsHead(nn.Module):
    def __init__(self, configer):
        super(HrtClsHead, self).__init__()
        self.pre_stage_channels = configer.in_channels
        self.num_classes = configer.num_classes
        self.incre_modules, self.downsamp_modules, self.final_layer = self._make_head(
            self.pre_stage_channels
        )
        self.classifier = nn.Linear(2048, self.num_classes)
    def _make_head(self, pre_stage_channels):
        head_block = Bottleneck
        head_channels = [32, 128, 256, 256]
        Log.info("pre_stage_channels: {}".format(pre_stage_channels))
        Log.info("head_channels: {}".format(head_channels))
        # Increasing the #channels on each resolution
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            incre_module = self._make_layer(
                head_block, channels, head_channels[i], 1, stride=1
            )
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)

        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = head_channels[i]
            out_channels = head_channels[i + 1]
            downsamp_module = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=in_channels,
                ),
                nn.BatchNorm2d(in_channels, momentum=BN_MOMENTUM),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=False),
            )
            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=head_channels[3],
                out_channels=2048,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(2048, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=False),
        )

        return incre_modules, downsamp_modules, final_layer

    def _make_layer(
            self,
            block,
            inplanes,
            planes,
            blocks,
            input_resolution=None,
            num_heads=1,
            stride=1,
            window_size=7,
            halo_size=1,
            mlp_ratio=4.0,
            q_dilation=1,
            kv_dilation=1,
            sr_ratio=1,
            attn_type="msw",
    ):

        layers = []

        if isinstance(block, GeneralTransformerBlock):
            layers.append(
                block(
                    inplanes,
                    planes,
                    num_heads,
                    window_size,
                    halo_size,
                    mlp_ratio,
                    q_dilation,
                    kv_dilation,
                    sr_ratio,
                    attn_type,
                )
            )
        else:
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)
    def forward(self,y_list):
            y = self.incre_modules[0](y_list[0])

            for i in range(len(self.downsamp_modules)):
                y = self.incre_modules[i + 1](y_list[i + 1]) + self.downsamp_modules[i](y)

            y = self.final_layer(y)
            y = F.avg_pool2d(y, kernel_size=y.size()[2:]).view(y.size(0), -1)
            y = self.classifier(y)
            return y