from  torchvision import models
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.layers.fpn import FPN
from torchsummary import summary
from lib.models.layers.conv import BasicConv, BasicDeconv, ResBlock
BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01

class VGG16_FPN(nn.Module):
    def __init__(self,pretrained=True):
        super(VGG16_FPN, self).__init__()

        vgg = models.vgg16_bn(pretrained=pretrained)
        features = list(vgg.features.children())

        self.layer1 = nn.Sequential(*features[0:23])
        self.layer2 = nn.Sequential(*features[23:33]) #1/8
        self.layer3 = nn.Sequential(*features[33:43])

        in_channels = [256,512,512]
        self.neck_seg = FPN(in_channels, 128, len(in_channels))
        self.neck_reg = FPN(in_channels, 128, len(in_channels))

        self.loc_head = nn.Sequential(
            nn.Dropout2d(0.1),
            ResBlock(in_dim=384, out_dim=128, dilation=0, norm="bn"),

            nn.ConvTranspose2d(128, 64, 2, stride=2, padding=0, output_padding=0, bias=False),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, 2, stride=2, padding=0, output_padding=0, bias=False),
            nn.BatchNorm2d(16, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):


        fea = []
        x = self.layer1(x)
        fea.append(x)
        x = self.layer2(x)
        fea.append(x)
        x = self.layer3(x)
        fea.append(x)


        x = self.neck_reg(fea)
        # x =torch.cat([x[0],  F.interpolate(x[1],scale_factor=2,mode='bilinear',align_corners=True),
        #                F.interpolate(x[2],scale_factor=4, mode='bilinear',align_corners=True)], dim=1)

        all_pre_map = self.loc_head(x)



        return all_pre_map

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels,padding, kernel_size,stride=1,use_bn='none'):
        super(BasicConv, self).__init__()
        self.use_bn = use_bn
        if self.use_bn == 'bn':
            self.bn = nn.BatchNorm2d(out_channels)
        elif self.use_bn == 'in':
            self.bn = nn.InstanceNorm2d(out_channels)
        elif self.use_bn == 'none':
            self.bn = None
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,stride=stride,
                              padding=padding,bias=not self.bn)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return F.relu(x, inplace=True)
class BasicDeconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, use_bn='none'):
        super(BasicDeconv, self).__init__()
        self.use_bn = use_bn
        if self.use_bn == 'bn':
            self.bn = nn.BatchNorm2d(out_channels)
        elif self.use_bn == 'in':
            self.bn = nn.InstanceNorm2d(out_channels)
        elif self.use_bn == 'none':
            self.bn = None
        self.tconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=0,bias=not self.bn)

    def forward(self, x):
        # pdb.set_trace()
        x = self.tconv(x)
        if self.bn is not None:
            x = self.bn(x)
        return F.relu(x, inplace=True)

class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx,input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return input.clamp_min_(0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input =  torch.ones_like(input, dtype=input.dtype, device=input.device)
        grad_input[input < 0] = torch.exp(input[input<0])
        grad_input = grad_input*grad_output
        return grad_input
def get_seg_model(cfg):
    model = VGG16_FPN(pretrained=True)

    return model

if __name__ == "__main__":
    net = VGG16_FPN(pretrained=False).cuda()
    # net(torch.rand(1,3,128,128).cuda(2))
    print(net)
    summary(net,(3,64 ,64 ),batch_size=4)
    # net(torch.rand(1,3,64,64).cuda())
