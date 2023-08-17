import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.layers.conv import BasicConv, BasicDeconv, ResBlock
import logging
import os
BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01
logger = logging.getLogger(__name__)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class InterNet(nn.Module):
    def __init__(self):
        super(InterNet, self).__init__()
        cfgs = {
            'stem':{
                'down1':{'layer_num': 1, 'in_chanel': 3, 'out_chanel':64},
                'down2': {'layer_num': 3, 'in_chanel': 64, 'out_chanel': 64},
                    },
            'floor_x4':{
                'stage1':{'layer_num': 4, 'in_chanel': 64, 'out_chanel':64},
                'stage2':{'layer_num': 4, 'in_chanel': 64, 'out_chanel': 64},
                'stage3':{'layer_num': 4, 'in_chanel': 64, 'out_chanel': 64},
                'stage4': {'layer_num':4, 'in_chanel': 64, 'out_chanel': 64}

            },
            'floor_x8': {
                'down3': {'layer_num': 1, 'in_chanel': 64, 'out_chanel': 128},
                'stage1': {'layer_num': 3, 'in_chanel': 128, 'out_chanel': 128},
                'stage2': {'layer_num': 4, 'in_chanel': 128, 'out_chanel': 128},
                'stage3': {'layer_num': 4, 'in_chanel': 128, 'out_chanel': 128},
                'stage4': {'layer_num': 4, 'in_chanel': 128, 'out_chanel': 128}

            },
            'floor_x16': {
                'down4': {'layer_num': 1, 'in_chanel': 128, 'out_chanel': 256},
                'stage1': {'layer_num': 2, 'in_chanel': 256, 'out_chanel': 256},
                'stage2': {'layer_num': 4, 'in_chanel': 256, 'out_chanel': 256},
                'stage3': {'layer_num': 4, 'in_chanel': 256, 'out_chanel': 256},
                'stage4': {'layer_num': 4, 'in_chanel': 256, 'out_chanel': 256}

            },
            'floor_x32': {
                'down5': {'layer_num': 1, 'in_chanel': 256, 'out_chanel': 256},
                'stage1': {'layer_num': 1, 'in_chanel': 256, 'out_chanel': 256},
                'stage2': {'layer_num': 4, 'in_chanel': 256, 'out_chanel': 256},
                'stage3': {'layer_num': 4, 'in_chanel': 256, 'out_chanel': 256},
                'stage4': {'layer_num': 4, 'in_chanel': 256, 'out_chanel': 256}

            },
        }

        self.stem = nn.Sequential(
            make_layers(BasicResidual, cfgs['stem']['down1'], down_sample=True),
            make_layers(BasicResidual, cfgs['stem']['down2'], down_sample=True)
                                  )

        self.floor1_1 = make_layers(BasicResidual, cfgs['floor_x4']['stage1'])
        self.floor1_2 = make_layers(BasicResidual, cfgs['floor_x4']['stage2'])
        self.floor1_3 = make_layers(BasicResidual,cfgs['floor_x4']['stage3'])
        self.floor1_4 = make_layers(BasicResidual,cfgs['floor_x4']['stage4'])

        self.floor2_d = make_layers(BasicResidual, cfgs['floor_x8']['down3'], down_sample=True)
        self.floor2_1 = make_layers(BasicResidual, cfgs['floor_x8']['stage1'])
        self.floor2_2 = make_layers(BasicResidual, cfgs['floor_x8']['stage2'])
        self.floor2_3 = make_layers(BasicResidual,cfgs['floor_x8']['stage3'])
        self.floor2_4 = make_layers(BasicResidual,cfgs['floor_x8']['stage4'])

        self.floor3_d = make_layers(BasicResidual, cfgs['floor_x16']['down4'],down_sample=True)
        self.floor3_1 = make_layers(BasicResidual, cfgs['floor_x16']['stage1'])
        self.floor3_2 = make_layers(BasicResidual, cfgs['floor_x16']['stage2'])
        self.floor3_3 = make_layers(BasicResidual,cfgs['floor_x16']['stage3'])
        self.floor3_4 = make_layers(BasicResidual,cfgs['floor_x16']['stage4'])

        self.floor4_d = make_layers(BasicResidual, cfgs['floor_x32']['down5'],down_sample=True)
        self.floor4_1 = make_layers(BasicResidual, cfgs['floor_x32']['stage1'])
        self.floor4_2 = make_layers(BasicResidual, cfgs['floor_x32']['stage2'])
        self.floor4_3 = make_layers(BasicResidual,cfgs['floor_x32']['stage3'])
        self.floor4_4 = make_layers(BasicResidual,cfgs['floor_x32']['stage4'])

        self.last_layer = nn.Sequential(
        # nn.Dropout2d(0.1),
        # ResBlock(in_dim=128, out_dim=256, dilation=0, norm="bn"),
        nn.Conv2d(704, 128, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(128, 128, 2, stride=2, padding=0, output_padding=0, bias=False),
        nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
        nn.ReLU(inplace=True),

        nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(32, momentum=BN_MOMENTUM),
        nn.ReLU(inplace=True),

        nn.ConvTranspose2d(32, 16, 2, stride=2, padding=0, output_padding=0, bias=False),
        nn.BatchNorm2d(16, momentum=BN_MOMENTUM),
        nn.ReLU(inplace=True),

        nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0,bias=False),
        nn.ReLU(inplace=True))
    def forward(self,x):
        x_main = self.stem(x)
        import pdb
        # pdb.set_trace()
        x1 = self.floor1_1(x_main)
        x1 = self.floor1_2(x1)
        x1 = self.floor1_3(x1)
        x1 = self.floor1_4(x1)

        x_main = self.floor2_d(x_main)
        x2 = self.floor2_1(x_main)
        x2 = self.floor2_2(x2)
        x2 = self.floor2_3(x2)
        x2 = self.floor2_4(x2)

        x_main = self.floor3_d(x_main)
        x3 = self.floor3_1(x_main)
        x3 = self.floor3_2(x3)
        x3 = self.floor3_3(x3)
        x3 = self.floor3_4(x3)

        x_main = self.floor4_d(x_main)
        x4 = self.floor3_1(x_main)
        x4 = self.floor3_2(x4)
        x4 = self.floor3_3(x4)
        x4 = self.floor3_4(x4)

        x0_h, x0_w = x1.size(2), x1.size(3)
        x2 = F.upsample(x2, size=(x0_h, x0_w), mode='bilinear')
        x3 = F.upsample(x3, size=(x0_h, x0_w), mode='bilinear')
        x4 = F.upsample(x4, size=(x0_h, x0_w), mode='bilinear')

        x = torch.cat([x1, x2, x3, x4], 1)

        x = self.last_layer(x)

        return x
    def init_weights(self, pretrained='',):
        # import pdb
        # pdb.set_trace()
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            #for k, _ in pretrained_dict.items():
            #    logger.info(
            #        '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
def make_layers(block, cfg, down_sample=False):
    layers = []
    for i in range(cfg['layer_num']):
        if i ==0:
            if down_sample:
                downsample = nn.Sequential(
                    nn.Conv2d(cfg['in_chanel'],  cfg['out_chanel'] * block.expansion,
                              kernel_size=1, stride=2, bias=False),
                    BatchNorm2d(cfg['out_chanel'] * block.expansion, momentum=BN_MOMENTUM),
                )
                stride=2

            else:
                downsample=None
                stride =1
            layers += [block(cfg['in_chanel'], cfg['out_chanel'], stride=stride,
                             downsample=downsample)]
        else:
            layers += [block(cfg['out_chanel'], cfg['out_chanel'], stride=1)]
    return nn.Sequential(*layers)

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels,stride=1):
        super(BasicConv, self).__init__()


        self.conv = conv3x3(in_channels, out_channels, stride=stride)
        self.norm = nn.BatchNorm2d(out_channels,eps=1e-05, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class BasicResidual(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicResidual, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleResidual(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleResidual, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def get_seg_model(cfg):
    model = InterNet()
    model.init_weights()

    return model
if __name__ == '__main__':
    from lib.utils.modelsummary import get_model_summary
    model = get_seg_model(3)
    dump_input = torch.rand(
        (1, 3,1024, 1024)
    )
    print(get_model_summary(model.to(1), dump_input.to(1)))