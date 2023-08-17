##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Donny You, RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.models.backbones import MocBackbone
from lib.models.backbones import MocHRBackbone
from lib.models.backbones import MocCatBackbone
from lib.models.backbones import MAEvitBackbone
from lib.models.backbones import VGGBackbone
from lib.models.backbones import HRBackboneFPN
from lib.utils.logger import Logger as Log

all_backbones = [
 
     "MocBackbone",
    "MocHRBackbone","MocCatBackbone", "MAEvitBackbone","VGGBackbone","HRBackboneFPN"
]

class BackboneSelector(object):
    def __init__(self, configer):
        self.configer = configer

    def get_backbone(self, **params):
        backbone = self.configer.backbone

        model = None
        if backbone in all_backbones:
            model=eval(backbone)(self.configer)(**params)

        else:
            Log.error("Backbone {} is invalid, the available backbones are one of those {}.".format(backbone, all_backbones))
            exit(1)

        return model
