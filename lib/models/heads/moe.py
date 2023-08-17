import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()


        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.soft(out)
        return out
class mask_generator(nn.Module):
    def __init__(self, in_channel, out_channel, patch_size):
        super(mask_generator, self).__init__()
        pooling_layer = []
        dst_patch_size = 8
        for i in range(2):
            pooling_layer.append(nn.Sequential(
                nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1,padding=1, bias=False),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(inplace=True),
            ))
        pooling_layer.append( nn.Sequential(
            nn.Conv2d(in_channel, 2, kernel_size=1),
            nn.Softmax(1)
            )
        )
        self.cls = nn.ModuleList(pooling_layer)

    def forward(self, x):
        for i in range(len(self.cls)):
            x = self.cls[i](x)


        return x

class MOE(nn.Module):
    """Image to Patch Embedding.

    We use a conv layer to implement PatchEmbed.

    Args:
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        conv_type (str): The type of convolution
            to generate patch embedding. Default: "Conv2d".
        kernel_size (int): The kernel_size of embedding conv. Default: 16.
        stride (int): The slide stride of embedding conv.
            Default: 16.
        padding (int | tuple | string): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int): The dilation rate of embedding conv. Default: 1.
        bias (bool): Bias of embed conv. Default: True.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        input_size (int | tuple | None): The size of input, which will be
            used to calculate the out size. Only works when `dynamic_size`
            is False. Default: None.
        init_cfg (`mmcv.ConfigDict`, optional): The Config for initialization.
            Default: None.
    """
    def __init__(self,
                 patch_size=(224, 224),
                 in_channels=3,
                 embed_dims=768,
                 conv_type='Conv2d',
                 kernel_size=16,
                 stride=16,
                 padding='corner',
                 dilation=1,
                 bias=True,
                 norm_cfg=None,
                 input_size=None,
                 init_cfg=None):
        super(MOE, self).__init__()
        self.patch_size = (patch_size[0]//4,patch_size[1]//4)
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.avg_kernel_size = (patch_size[0] // 2**8, patch_size[1]//2**8)
        self.conv = nn.Sequential(
            nn.Conv2d(48+96, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=1),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=1),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=1),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=1),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, stride=1,padding=1),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(128, 128, kernel_size=3, stride=1,padding=1,bias=False),
            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),
            nn.AvgPool2d(self.avg_kernel_size),
            nn.Conv2d(128, 4, kernel_size=1)

        )
        # self.mlp = MLP(1024,4, 512)

        self.init_weights()
    def init_weights(self, pretrained='',):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        x0_h, x0_w = input[0].size(2), input[0].size(3)
        x1 = F.upsample(input[1], size=(x0_h, x0_w), mode='nearest')
        # x2 = F.upsample(input[2], size=(x0_h, x0_w), mode='nearest')
        # x3 = F.upsample(input[3], size=(x0_h, x0_w), mode='nearest')
        input = torch.cat([input[0], x1], 1)
        input = input.detach()

        B, C, H,W = input.size()
        H_num = H//self.patch_size[0]
        W_num = W//self.patch_size[1]
        pre_slice = F.unfold(input, self.patch_size, stride=self.patch_size) # B, KK, L
        pre_slice = pre_slice.transpose(2,1) #B,L, CKK

        pre_slice = pre_slice.reshape(-1, pre_slice.size(-1))
        pre_slice = pre_slice.view(-1, C, self.patch_size[0], self.patch_size[1])

        x_mask =  self.conv(pre_slice)
        # x_mask = self.mlp(x.view(x.size(0),-1))
        x_mask = x_mask.flatten(start_dim=1)

        # import pdb
        # pdb.set_trace()
        x_mask = x_mask.view(B,-1,x_mask.size(-1)).transpose(1,2)
        x_mask = x_mask.view(B,x_mask.size(1),H_num,W_num)

        return x_mask


BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
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



class one_branch_output(nn.Module):
    def __init__(self, input_channels,
                 counter_inchannels,
                 tmp_patch_size,
                 low_or_high='medium',
                 fusion_method='cat'):
        super(one_branch_output, self).__init__()
        self.fusion_method = fusion_method
        if self.fusion_method == "cat":
            expect_channels = counter_inchannels//2
        elif self.fusion_method == "sum":
            expect_channels = counter_inchannels
        else:
            raise ValueError("Unknown fusion method")

        if input_channels==expect_channels:
            self.channels_down = nn.Identity()

        else:
            if low_or_high == 'low':
                expect_channels = counter_inchannels
            self.channels_down = nn.Sequential(
                nn.Conv2d( input_channels, expect_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(expect_channels),
                nn.ReLU(inplace=True)
            )

        if low_or_high == 'low':
            self.modulation_layer = nn.Identity()

        else:
            self.modulation_layer_small = nn.Sequential(
                nn.Conv2d(counter_inchannels, expect_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(expect_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(expect_channels, expect_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(expect_channels),
                nn.ReLU(inplace=True),
            )

            self.modulation_layer_big = nn.Sequential(
                nn.Conv2d(counter_inchannels, expect_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(expect_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(expect_channels, expect_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(expect_channels),
                nn.ReLU(inplace=True),
            )
            self.soft_mask = mask_generator(counter_inchannels, out_channel=2, patch_size=tmp_patch_size)
        # self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, current_stage, last_stage=None, counter=None,out_branch=False,no_distangle=False):
        if no_distangle:
            current_stage = self.channels_down(current_stage)
            if last_stage is None:
                current_stage = current_stage

            else:
                last_stage = F.interpolate(last_stage, scale_factor=2,mode='nearest')
                last_stage = self.modulation_layer_big(last_stage)

                if self.fusion_method == "cat":
                    current_stage = torch.cat([current_stage, last_stage],1)
                elif self.fusion_method == "sum":
                    current_stage = current_stage+last_stage

            out_put = counter(current_stage)

            return out_put, current_stage
        else:
            current_stage = self.channels_down(current_stage)
            if last_stage is None:
                current_stage = current_stage
                out_put = counter(current_stage)
                # import numpy as np

                # np.save('./exp/moe/current_{}.npy'.format(current_stage.size(2)),current_stage.cpu().numpy())
            else:
                if out_branch:
                    # mask = self.soft_mask(last_stage)
                    # mask = F.interpolate(mask, scale_factor=2,mode='nearest')
                    # import numpy as np
                    # import cv2
                    # pred_color_map= mask[:,0,:,:].cpu().numpy()
                    # pred_color_map = cv2.applyColorMap(
                    #     (255 * pred_color_map / (pred_color_map.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
                    # cv2.imwrite('./exp/moe/first_mask0_{}.png'.format(mask.size(2)), pred_color_map)
                    #
                    # pred_color_map= mask[:,1,:,:].cpu().numpy()
                    # pred_color_map = cv2.applyColorMap(
                    #     (255 * pred_color_map / (pred_color_map.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
                    # cv2.imwrite('./exp/moe/first_mask1_{}.png'.format(mask.size(2)), pred_color_map)
                    #
                    # import pdb
                    # pdb.set_trace()
                    last_stage = F.interpolate(last_stage, scale_factor=2,mode='nearest')
                    last_stage = self.modulation_layer_big(last_stage)
                    # last_stage = last_stage*mask[:,1,:,:].unsqueeze(1)
                    current_stage = torch.cat([current_stage, last_stage],1)
                    out_put = counter(current_stage)
                else:
                    mask = self.soft_mask(last_stage)
                    mask = F.interpolate(mask, scale_factor=2,mode='nearest')
                    #
                    # import numpy as np
                    # np.save('./exp/moe/{}.npy'.format(mask.size(2)),mask.cpu().numpy())
                    # np.save('./exp/moe/current_{}.npy'.format(current_stage.size(2)),current_stage.cpu().numpy())

                    last_stage = F.interpolate(last_stage, scale_factor=2,mode='nearest')


                    last_stage_small = self.modulation_layer_small(last_stage)
                    last_stage_large = self.modulation_layer_big(last_stage)

                    last_stage_small = last_stage_small*mask[:,0,:,:].unsqueeze(1)
                    last_stage_large = last_stage_large*mask[:,1,:,:].unsqueeze(1)

                    out_put = counter(torch.cat([current_stage, last_stage_large],1))

                    last_stage = last_stage_small+last_stage_large

                    if self.fusion_method == "cat":
                        current_stage = torch.cat([current_stage, last_stage],1)
                    elif self.fusion_method == "sum":
                        current_stage = current_stage+last_stage


            return out_put, current_stage

class upsample_module(nn.Module):
    def __init__(self,
                 config
          ):
        super(upsample_module, self).__init__()
        self.config = config
        self.stages_channel = config.stages_channel
        self.fuse_method = config.fuse_method
        self.num_resolutions = len(self.stages_channel)
        self.counter_inchannels = config.in_channels

        output_heads= []
        for i in range(self.num_resolutions):
            if i == 0:
                low_or_high = 'low'
            elif i == self.num_resolutions-1:
                low_or_high ='high'
            else:
                low_or_high = 'medium'
            output_heads.append(
                one_branch_output(self.stages_channel[i],
                                  self.counter_inchannels,
                                  tmp_patch_size=8*(2**i),
                                  low_or_high=low_or_high,
                                  fusion_method=self.fuse_method))
        self.multi_outputs = nn.ModuleList(output_heads)


    def forward(self,in_list, counter, counter_copy):

        assert  len(in_list) == self.num_resolutions
        out_list = []


        output, last_stage =  self.multi_outputs[0](in_list[-1],
                                                    last_stage=None,
                                                    counter=counter_copy)
        # import numpy as np
        # import cv2
        # pred_color_map= output[:,:,10:,:].cpu().numpy()
        # pred_color_map = cv2.applyColorMap(
        #     (255 * pred_color_map / (pred_color_map.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        # cv2.imwrite('./exp/moe/density_{}.png'.format(output.size(2)), pred_color_map)

        out_list.append(output)
        for i in range(1,self.num_resolutions):
            if i <self.num_resolutions-1:
                output, last_stage = self.multi_outputs[i](in_list[-(i+1)],
                                                           last_stage, counter_copy)
            else:
                output, last_stage = self.multi_outputs[i](in_list[-(i+1)],last_stage,
                                                           counter,out_branch=True)
            out_list.insert(0,output)
            # import numpy as np
            # import cv2
            # pred_color_map= output[:,:,10*2**i:,:].cpu().numpy()
            # pred_color_map = cv2.applyColorMap(
            #     (255 * pred_color_map / (pred_color_map.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
            # cv2.imwrite('./exp/moe/density_{}.png'.format(output.size(2)), pred_color_map)

        return out_list

class FusionBYconv(nn.Module):
    """Image to Patch Embedding.

    We use a conv layer to implement PatchEmbed.

    Args:
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        conv_type (str): The type of convolution
            to generate patch embedding. Default: "Conv2d".
        kernel_size (int): The kernel_size of embedding conv. Default: 16.
        stride (int): The slide stride of embedding conv.
            Default: 16.
        padding (int | tuple | string): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int): The dilation rate of embedding conv. Default: 1.
        bias (bool): Bias of embed conv. Default: True.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        input_size (int | tuple | None): The size of input, which will be
            used to calculate the out size. Only works when `dynamic_size`
            is False. Default: None.
        init_cfg (`mmcv.ConfigDict`, optional): The Config for initialization.
            Default: None.
    """
    def __init__(self,
                 config):
        super(FusionBYconv, self).__init__()

        self.pre_stage_channels = config.in_channels
        self.fuse_method = config.fuse_method
        self.upsamp_modules = self._make_head(
            self.pre_stage_channels
        )

        self.init_weights()

    def _make_head(self, pre_stage_channels):
        # downsampling modules
        upsamp_modules = []
        for i in range(len(pre_stage_channels)-1):
            in_channels =  self.pre_stage_channels[i]
            out_channels =  self.pre_stage_channels[i+1]
            downsamp_module = nn.Sequential(
                # nn.Upsample(scale_factor=2,mode="bilinear"),
                nn.Conv2d(in_channels+out_channels, out_channels, kernel_size=3, stride=1,padding=1, bias=False),
                BatchNorm2d(out_channels,momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            )
            upsamp_modules.append(downsamp_module)
        upsamp_modules = nn.ModuleList(upsamp_modules)

        return upsamp_modules

    def init_weights(self, pretrained=''):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,in_list):
        x0_h, x0_w = in_list[0].size(2), in_list[0].size(3)
        for i in range(1, len(in_list),1):
            in_list[i] = F.upsample(in_list[i], size=(x0_h, x0_w), mode='bilinear')
        y = torch.cat(in_list, 1)

        return y