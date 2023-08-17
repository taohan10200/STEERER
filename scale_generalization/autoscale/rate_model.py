# coding=utf-8
import torch.nn as nn
import torch
from torchvision import models
from scale_generalization.autoscale.utils import save_net, load_net
import torch.nn.functional as F


class RATEnet(nn.Module):
    def __init__(self, load_weights=False):
        super(RATEnet, self).__init__()
        self.seen = 0

        self.des_dimension = nn.Sequential(
            nn.Conv2d(152,64,3,padding=1),
            nn.ReLU(inplace=True),
            )

        self.ROI_feat = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

                                      nn.Conv2d(32, 32, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                                      )
        self.output = nn.Sequential(
                                      nn.Linear(32*3*3, 1),
                                      # nn.ReLU(inplace=True),
                                      #
                                      # nn.Linear(10, 1)
        )

        #
        # # self.backend = make_layers(self.backend_feat,in_channels = 512,dilation = True)
        # # self.upscore = nn.UpsamplingBilinear2d(scale_factor=8)
        # self.output_layer = nn.Sequential(
        #     nn.UpsamplingBilinear2d(scale_factor=2),
        #     nn.Conv2d(64, 11, 1),
        # )
        # if not load_weights:
        #     mod = models.vgg16(pretrained = True)
        self._initialize_weights()
        # for i in xrange(len(self.frontend.state_dict().items())):
        #     self.frontend.state_dict().items()[i][1].data[:] = mod.state_dict().items()[i][1].data[:]

    def forward(self, x):

        x = self.des_dimension(x)
        x = self.ROI_feat(x)
        #print("x.size", x.size)
        x = x.view(x.size(0), 32*3*3)
        x =torch.abs(self.output(x))
        #x = 0.5+2*F.sigmoid(x)

        return x

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.normal_(m.weight, mean=0.2,std=0.1)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
