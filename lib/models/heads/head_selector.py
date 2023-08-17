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

from lib.utils.logger import Logger as Log
from lib.models.heads import *

all_heads = [
    'ClsHead', 'LinearClsHead', 'StackedLinearClsHead', 'MultiLabelClsHead',
    'MultiLabelLinearClsHead', 'VisionTransformerClsHead', 'DeiTClsHead',
    'ConformerHead','CountingHead','HrtClsHead','MocClsHead', 'LocalizationHead'
]

class HeadSelector(object):
    def __init__(self, configer):
        self.configer = configer

    def get_head(self, **params):
        head = self.configer.type

        if head in all_heads:
            model =  eval(head)(self.configer)
        else:
            Log.error("Backbone {} is invalid, the available heads are {} ".format(head, all_heads))
            exit(1)

        return model

if __name__ == "__main__":
    network=\
        dict(head='CountingHead')
    print(network)
    model = HeadSelector(network).get_head()
    print(model)
