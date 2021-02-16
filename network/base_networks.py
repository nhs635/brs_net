
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
                 kernel_size=(3, 3), padding_type="zero", padding=(1, 1), dilation=(1, 1), stride=(1, 1),
                 norm_layer=nn.BatchNorm2d, activation=nn.ReLU, preactivation=False):
        super(ConvBlock, self).__init__()

        # use_bias setup
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.conv_block = []

        # Normalization & activation
        if preactivation:
            self.conv_block += [norm_layer(num_features=in_channels)] if norm_layer is not None else []
            self.conv_block += [activation(inplace=True)] if activation is not None else []

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

        # Convolution
        self.conv_block += [nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=p,
                                      dilation=dilation,
                                      bias=use_bias)]

        # Normalization & activation
        if not preactivation:
            self.conv_block += [norm_layer(num_features=out_channels)] if norm_layer is not None else []
            self.conv_block += [activation(inplace=True)] if activation is not None else []

        self.conv_block = nn.Sequential(*self.conv_block)

    def forward(self, x):
        return self.conv_block(x)


class ConvBlock2(nn.Module):
    """ Define a double convolution block (conv + norm + actv). """
    def __init__(self, in_channels, out_channels,
                 kernel_size=(3, 3), padding_type="zero", padding=(1, 1), dilation=(1, 1), stride=(1, 1),
                 norm_layer=nn.BatchNorm2d, activation=nn.ReLU):
        super(ConvBlock2, self).__init__()

        self.conv_block = []
        self.conv_block += [ConvBlock(in_channels=in_channels, out_channels=out_channels,
                                      kernel_size=kernel_size, padding_type=padding_type, padding=padding,
                                      dilation=dilation, stride=(1, 1),
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
                 kernel_size=(3, 3), padding=(1, 1), output_padding=(0, 0), dilation=(1, 1), stride=(1, 1),
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


class ResidualBlock(nn.Module):
    """ Define a residual network block. """
    def __init__(self, in_channels, out_channels, padding_type="zero", downsample=False,
                 norm_layer=nn.BatchNorm2d, activation=nn.ReLU, preactivation=False):
        """ Initialize a residual network block """
        super(ResidualBlock, self).__init__()

        # Residual block definition
        self.residual_block = []
        self.residual_block += [ConvBlock(in_channels, out_channels, stride=(1, 1) if not downsample else (2, 2),
                                          kernel_size=(3, 3), padding_type=padding_type,
                                          padding=(1, 1) if padding_type == "zero" else (1,)*4,
                                          norm_layer=norm_layer, activation=activation,
                                          preactivation=preactivation)]
        self.residual_block += [ConvBlock(out_channels, out_channels, stride=(1, 1),
                                          kernel_size=(3, 3), padding_type=padding_type,
                                          padding=(1, 1) if padding_type == "zero" else (1,)*4,
                                          norm_layer=norm_layer, activation=None,
                                          preactivation=preactivation)]
        self.residual_block = nn.Sequential(*self.residual_block)
        self.activation = activation()

        # Bypass 1x1 convolution
        self.conv_1x1 = []
        if (in_channels != out_channels) or downsample:
            if preactivation:
                self.conv_1x1 += [norm_layer(num_features=in_channels)] if norm_layer is not None else []
            self.conv_1x1 += [nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=(1, 1),
                                        stride=(1, 1) if not downsample else (2, 2),
                                        bias=False)]
            if not preactivation:
                self.conv_1x1 += [norm_layer(num_features=out_channels)] if norm_layer is not None else []
        else:
            self.conv_1x1 += [nn.LeakyReLU(negative_slope=1)]  # identity mapping

        self.conv_1x1 = nn.Sequential(*self.conv_1x1)

    def forward(self, x):
        """ Forward function with skip connection. """
        y = self.conv_1x1(x) + self.residual_block(x)  # add skip connection
        return self.activation(y)


class MixedPooling(nn.Module):
    """ Define a mixed pooling module. """
    def __init__(self):
        super(MixedPooling, self).__init__()

        self.avg_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.a = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=True)

        self.register_parameter(name="alpha", param=self.a)

    def forward(self, x):
        s = torch.sigmoid(self.a)
        return s * self.avg_pool(x) + torch.sub(1, s) * self.max_pool(x)


class Attention(nn.Module):
    """ Define an (channel + spatial) attention block. """
    def __init__(self, in_channels, reduction_ratio=1):
        """ Initialize an attention block """
        super(Attention, self).__init__()

        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.gmp = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        self.fc1 = nn.Linear(in_features=in_channels, out_features=int(in_channels/reduction_ratio))
        self.fc2 = nn.Linear(in_features=int(in_channels/reduction_ratio), out_features=in_channels)
        self.conv = nn.Conv2d(in_channels=2, out_channels=in_channels,
                              kernel_size=(7, 7), padding=(3, 3))

    def forward(self, x):
        # Channel attention module
        a_c = torch.sigmoid(self.fc2(self.fc1(torch.squeeze(self.gap(x)))) +
                            self.fc2(self.fc1(torch.squeeze(self.gmp(x)))))
        y = torch.mul(x, a_c.unsqueeze(2).unsqueeze(3).repeat(1, 1, x.size(2), x.size(3)))

        # Spatial attention module
        a_s = torch.sigmoid(self.conv(torch.cat((torch.max(y, dim=1, keepdim=True)[0],
                                                 torch.mean(y, dim=1, keepdim=True)), dim=1)))
        y = torch.mul(y, a_s)

        return y


class DilatedModule(nn.Module):
    """ Define a dilated convolution module block. """
    def __init__(self, in_channels, intmed_channels):
        """ Initialize a dilated conv module block """
        super(DilatedModule, self).__init__()

        self.conv_1x1_r = nn.Conv2d(in_channels=in_channels, out_channels=intmed_channels, kernel_size=(1, 1))
        self.conv_d1 = nn.Conv2d(in_channels=intmed_channels, out_channels=intmed_channels,
                                 kernel_size=(3, 3), padding=1, dilation=1)
        self.conv_d2 = nn.Conv2d(in_channels=intmed_channels, out_channels=intmed_channels,
                                 kernel_size=(3, 3), padding=2, dilation=2)
        self.conv_d3 = nn.Conv2d(in_channels=intmed_channels, out_channels=intmed_channels,
                                 kernel_size=(3, 3), padding=3, dilation=3)
        self.conv_1x1_a = nn.Conv2d(in_channels=3 * intmed_channels, out_channels=in_channels, kernel_size=(1, 1))

    def forward(self, x):
        x = self.conv_1x1_r(x)
        x1 = self.conv_d1(x)
        x2 = self.conv_d2(x)
        x3 = self.conv_d3(x)
        return self.conv_1x1_a(torch.cat((x1, x2, x3), dim=1))


class GlobalConvModule(nn.Module):
    """ Define a global convolution module block (with boundary refinement). """
    def __init__(self, in_channels, out_channels, kernel_size=15):
        """ Initialize a global conv module block """
        super(GlobalConvModule, self).__init__()

        k = kernel_size
        p = int((k - 1) / 2)
        self.conv_kx1_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=(k, 1), padding=(p, 0))
        self.conv_1xk_1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                    kernel_size=(1, k), padding=(0, p))
        self.conv_1xk_2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=(1, k), padding=(0, p))
        self.conv_kx1_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                    kernel_size=(k, 1), padding=(p, 0))

        self.br = [nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                             kernel_size=(3, 3), padding=(1, 1)),
                   nn.ReLU(),
                   nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                             kernel_size=(3, 3), padding=(1, 1))]
        self.br = nn.Sequential(*self.br)

    def forward(self, x):
        # global convolutional network
        x1 = self.conv_1xk_1(self.conv_kx1_1(x))
        x2 = self.conv_kx1_2(self.conv_1xk_2(x))
        y = x1 + x2

        # boundary refinement
        y = y + self.br(y)

        return y
