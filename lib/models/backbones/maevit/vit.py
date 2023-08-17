# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
import logging
logger = logging.getLogger(__name__)

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__( **kwargs)
        self.patch_size = kwargs['patch_size']
        # import pdb
        # pdb.set_trace()
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B, _, H, W = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        # ============My adding code==========
        x = x[:, 1:, :]   #.mean(dim=1)


        patch_h, patch_w = H//self.patch_size, W//self.patch_size
        x = x.permute(0, 2, 1).view(B, -1, patch_h, patch_w)
        # import pdb
        # pdb.set_trace()

        return x
        # ============My adding code==========


        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome
    def forward(self, x):
        x = self.forward_features(x)

        return [x]
    def init_weights(self, pretrained='',):
        import os
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)['model']

            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()


            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys() and "pos_embed" not in k}

            #for k, _ in pretrained_dict.items():
            #    logger.info(
            #        '=> loading {} pretrained model {}'.format(k, pretrained))
            logger.info(
                "Missing keys: {}".format(list(set(model_dict) - set(pretrained_dict)
                                               )))
            model_dict.update(pretrained_dict)
            # import pdb
            # pdb.set_trace()
            self.load_state_dict(model_dict)

def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        img_size=(768, 1024), patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


class MAEBackbone(object):
    def __init__(self, configer):
        self.configer = configer

    def __call__(self):
        arch = self.configer.sub_arch
        from lib.models.backbones.hrt.hrt_config import MODEL_CONFIGS

        if arch == "vit_base":
            arch_net = vit_base_patch16()
            arch_net.init_weights(self.configer.pretrained_backbone)
        elif arch == "vit_large":
            arch_net = vit_large_patch16()
            arch_net.load_pretrained(self.configer.pretrained_backbone)
        elif arch == "vit_huge":
            arch_net =  vit_huge_patch14()
            arch_net.load_pretrained(self.configer.pretrained_backbone)
            # arch_net = ModuleHelper.load_model(
            #     arch_net,
            #     pretrained=self.configer.pretrained_backbone,
            #     all_match=False,
            #     network="hrt_window" if "win" in arch else "hrt",
            # )

        else:
            raise Exception("Architecture undefined!")

        return arch_net