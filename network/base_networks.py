
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools


def init_weights(net, init_type="kaiming", init_gain=0.02):
    """ Initialize network weights.
    Parameters:
        net (network): network to be initialized
        init_type (str): the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float): scaling factor for normal, xavier and orthogonal.
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError("initialization method [%s] is not implemented" % init_type)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    # print("[%s] Initialize network with %s." % (net.__name__(), init_type))
    net.apply(init_func)


def init_net(net, init_type="normal", init_gain=0.02, device_ids=None):
    """ Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network): the network to be initialized
        init_type (str): the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float): scaling factor for normal, xavier and orthogonal.
        device_ids (int list): which GPUs the network runs on: e.g., 0, 1, 2
    Return an initialized network.
    """
    if torch.cuda.is_available():
        if device_ids is None:
            device_ids = []
        net.to(device_ids[0])
        if len(device_ids) > 1:
            net = nn.DataParallel(net, device_ids=device_ids)  # multi-GPUs
    else:
        net.to("cpu")
    init_weights(net, init_type=init_type, init_gain=init_gain)  # Weight initialization

    return net


def set_requires_grad(nets, requires_grad=False):
    """ Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list): a list of networks
        requires_grad (bool): whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class ConvBlock(nn.Module):
    """ Define a convolution block (conv + norm + actv). """
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, padding_type="zero", padding=1, dilation=1, stride=1,
                 norm_layer=nn.BatchNorm2d, activation=nn.ReLU):
        super(ConvBlock, self).__init__()

        # use_bias setup
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.conv_block = []
        # Padding option
        p = 0
        if padding_type == "reflect":
            self.conv_block += [nn.ReflectionPad2d(padding)]
        elif padding_type == "replicate":
            self.conv_block += [nn.ReplicationPad2d(padding)]
        elif padding_type == "zero":
            p = padding
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        self.conv_block += [nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=p,
                                      dilation=dilation,
                                      bias=use_bias)]
        self.conv_block += [norm_layer(num_features=out_channels)] if norm_layer is not None else []
        self.conv_block += [activation(inplace=True)] if activation is not None else []
        self.conv_block = nn.Sequential(*self.conv_block)

    def forward(self, x):
        return self.conv_block(x)


class ConvBlock2(nn.Module):
    """ Define a double convolution block (conv + norm + actv). """
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, padding_type="zero", padding=1, dilation=1, stride=1,
                 norm_layer=nn.BatchNorm2d, activation=nn.ReLU):
        super(ConvBlock2, self).__init__()

        self.conv_block = []
        self.conv_block += [ConvBlock(in_channels=in_channels, out_channels=out_channels,
                                      kernel_size=kernel_size, padding_type=padding_type, padding=padding,
                                      dilation=dilation, stride=1,
                                      norm_layer=norm_layer, activation=activation),
                            ConvBlock(in_channels=out_channels, out_channels=out_channels,
                                      kernel_size=kernel_size, padding_type=padding_type, padding=padding,
                                      dilation=dilation, stride=stride,
                                      norm_layer=norm_layer, activation=activation)]

        self.conv_block = nn.Sequential(*self.conv_block)

    def forward(self, x):
        return self.conv_block(x)


class UpConv(nn.Module):
    """ Define a convolution block with upsampling. """
    def __init__(self, in_channels, out_channels, scale_factor=2,
                 kernel_size=(3, 3), padding=(1, 1),
                 mode="nearest"):
        super(UpConv, self).__init__()

        self.up_conv = [nn.Upsample(scale_factor=scale_factor, mode=mode),
                        ConvBlock(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=kernel_size)]
        self.up_conv = nn.Sequential(*self.up_conv)

    def forward(self, x):
        return self.up_conv(x)


class DeconvBlock(nn.Module):
    """ Define a deconvolution block (deconv + norm + actv). """
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, padding=1, output_padding=0, dilation=1, stride=1,
                 norm_layer=nn.BatchNorm2d, activation=nn.ReLU):
        super(DeconvBlock, self).__init__()

        # use_bias setup
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.deconv_block = []
        self.deconv_block += [nn.ConvTranspose2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding,
                                                 output_padding=output_padding,
                                                 dilation=dilation,
                                                 bias=use_bias)]
        self.deconv_block += [norm_layer(num_features=out_channels)] if norm_layer is not None else []
        self.deconv_block += [activation(inplace=True)] if activation is not None else []
        self.deconv_block = nn.Sequential(*self.deconv_block)

    def forward(self, x):
        return self.deconv_block(x)


class SubPixelConv(nn.Module):
    def __init__(self, num_features, scale_factor=2, kernel_size=3):
        super(SubPixelConv, self).__init__()

        self.subpixel_conv = [ConvBlock(num_features, num_features * (scale_factor ** 2),
                                        kernel_size=kernel_size,
                                        padding_type="reflect", padding=int((kernel_size - 1) / 2.0),
                                        norm_layer=None, activation=nn.ReLU),
                              nn.PixelShuffle(upscale_factor=scale_factor)]
        self.subpixel_conv = nn.Sequential(*self.subpixel_conv)

    def forward(self, x):
        return self.subpixel_conv(x)


class ResnetBlock(nn.Module):
    """ Define a residual network block. """
    def __init__(self, num_features, padding_type="zero", norm_layer=nn.BatchNorm2d):
        """ Initialize a residual network block """
        super(ResnetBlock, self).__init__()

        # use_bias setup
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.resnet_block = []
        for i in range(2):
            # Convolution block
            self.resnet_block += [ConvBlock(num_features, num_features, kernel_size=3,
                                            padding_type=padding_type, padding=1,
                                            norm_layer=norm_layer, activation=nn.ReLU if i == 0 else None)]
        self.resnet_block = nn.Sequential(*self.resnet_block)

    def forward(self, x):
        """ Forward function with skip connection. """
        y = x + self.resnet_block(x)  # add skip connection
        return F.relu(y, inplace=True)
