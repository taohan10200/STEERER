import torch.nn as nn
import torch.nn.functional as F
# from .conv_ws import ConvWS2d, ConvAWS2d



conv_cfg = {
    'Conv': nn.Conv2d,
    # 'ConvWS': ConvWS2d,
    # 'ConvAWS': ConvAWS2d,
    # TODO: octave conv
}


class BasicDeconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, activate=None):
        super(BasicDeconv, self).__init__()
        bias = False if activate == 'bn' else True
        self.tconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=not self.use_bn)
        if activate == 'bn':
            self.bn = nn.BatchNorm2d(out_channels)
        elif activate == 'in':
            self.bn = nn.InstanceNorm2d(out_channels)
        elif activate == None:
            self.bn = None
    def forward(self, x):
        # pdb.set_trace()
        x = self.tconv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size,stride=1, padding=0,dilation=1, norm=None, relu =False):
        super(BasicConv, self).__init__()
        self.relu = relu
        bias = True if  norm is None else  False
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,stride=stride,
                              padding=padding,dilation=dilation, bias=bias)
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels,eps=1e-05, momentum=0.01)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == None:
            self.norm = None


    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x) if self.norm is  not None else x
        x = F.relu(x, inplace=True) if self.relu else x
        return x

class ResBlock(nn.Module):
    def __init__(self, in_dim,out_dim, dilation=1,  norm="bn"):
        super(ResBlock, self).__init__()
        padding = dilation+1
        model = []
        medium_dim = in_dim//4
        model.append(BasicConv(in_dim, medium_dim, 1, 1, 0, norm = norm, relu =True))
        model.append(BasicConv(medium_dim, medium_dim, 3, 1, padding = padding, dilation=dilation+1, norm=norm,  relu =True))
        model.append(BasicConv(medium_dim, out_dim, 1, 1, 0, norm=norm, relu =False))
        self.model = nn.Sequential(*model)
        if in_dim !=out_dim:
            self.downsample =  BasicConv(in_dim, out_dim, 1, 1, 0, norm=norm, relu =False)
        else:
            self.downsample =None
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x
        out = self.model(x)
        if self.downsample  is not None:

            out += self.downsample(residual)
        else:
            out += residual
        out = self.relu(out)
        return out
def build_conv_layer(cfg, *args, **kwargs):
    """ Build convolution layer

    Args:
        cfg (None or dict): cfg should contain:
            type (str): identify conv layer type.
            layer args: args needed to instantiate a conv layer.

    Returns:
        layer (nn.Module): created conv layer
    """
    if cfg is None:
        cfg_ = dict(type='Conv')
    else:
        assert isinstance(cfg, dict) and 'type' in cfg
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in conv_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        conv_layer = conv_cfg[layer_type]

    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer
