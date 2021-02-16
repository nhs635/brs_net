
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

from .base_networks import *


class GANLoss(nn.Module):
    """ Define various GAN objectives. """
    def __init__(self, gan_mode):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str): the type of GAN objectives (vanilla, lsgan, wgangp)
        Note: the discriminator should not return the sigmoid output.
        """
        super(GANLoss, self).__init__()
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ["wgangp"]:
            self.loss = None
        else:
            raise NotImplementedError("gan mode %s not implemented" % gan_mode)

    def __call__(self, prediction, is_target_real):
        """ Calculate loss given dicriminator's output and ground truth labels.
        Parameters:
            prediction (tensor): discriminator's output
            is_target_real (bool): real or fake?
        Returns:
            the calculated loss
        """
        if self.gan_mode in ["lsgan", "vanilla"]:
            label = torch.ones if is_target_real else torch.zeros
            target_tensor = label(prediction.shape, dtype=prediction.dtype).to(prediction.device)
            loss = self.loss(prediction, target_tensor)
        else:
            loss = -prediction.mean() if is_target_real else prediction.mean()

        return loss


class UNetGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(UNetGenerator, self).__init__()

        # Encoding paths
        self.enc_conv1 = ConvBlock(in_channels, 64, kernel_size=4, stride=2, padding_type="reflect", padding=1,
                                   norm_layer=None, activation=None)
        self.enc_conv2 = ConvBlock(64, 128, kernel_size=4, stride=2, padding_type="reflect", padding=1,
                                   norm_layer=norm_layer, activation=None)
        self.enc_conv3 = ConvBlock(128, 256, kernel_size=4, stride=2, padding_type="reflect", padding=1,
                                   norm_layer=norm_layer, activation=None)
        self.enc_conv4 = ConvBlock(256, 512, kernel_size=4, stride=2, padding_type="reflect", padding=1,
                                   norm_layer=norm_layer, activation=None)
        self.enc_conv5 = ConvBlock(512, 512, kernel_size=4, stride=2, padding_type="reflect", padding=1,
                                   norm_layer=norm_layer, activation=None)
        self.enc_conv6 = ConvBlock(512, 512, kernel_size=4, stride=2, padding_type="reflect", padding=1,
                                   norm_layer=norm_layer, activation=None)
        self.enc_conv7 = ConvBlock(512, 512, kernel_size=4, stride=2, padding_type="reflect", padding=1,
                                   norm_layer=norm_layer, activation=None)
        self.enc_conv8 = ConvBlock(512, 512, kernel_size=4, stride=2, padding_type="reflect", padding=1,
                                   norm_layer=norm_layer, activation=nn.ReLU)

        # Decoding paths
        self.dec_conv8 = DeconvBlock(512, 512, kernel_size=4, stride=2, padding=1,
                                     norm_layer=norm_layer, activation=None)
        self.dec_conv7 = DeconvBlock(1024, 512, kernel_size=4, stride=2, padding=1,
                                     norm_layer=norm_layer, activation=None)
        self.dec_conv6 = DeconvBlock(1024, 512, kernel_size=4, stride=2, padding=1,
                                     norm_layer=norm_layer, activation=None)
        self.dec_conv5 = DeconvBlock(1024, 512, kernel_size=4, stride=2, padding=1,
                                     norm_layer=norm_layer, activation=nn.ReLU)
        self.dec_conv4 = DeconvBlock(1024, 256, kernel_size=4, stride=2, padding=1,
                                     norm_layer=norm_layer, activation=nn.ReLU)
        self.dec_conv3 = DeconvBlock(512, 128, kernel_size=4, stride=2, padding=1,
                                     norm_layer=norm_layer, activation=nn.ReLU)
        self.dec_conv2 = DeconvBlock(256, 64, kernel_size=4, stride=2, padding=1,
                                     norm_layer=norm_layer, activation=nn.ReLU)
        self.dec_conv1 = DeconvBlock(128, out_channels, kernel_size=4, stride=2, padding=1,
                                     norm_layer=None, activation=None)

    def forward(self, x, leaky_alpha=0.2):
        out = dict()

        out["e1"] = F.leaky_relu(self.enc_conv1(x), leaky_alpha)
        out["e2"] = F.leaky_relu(self.enc_conv2(out["e1"]), leaky_alpha)
        out["e3"] = F.leaky_relu(self.enc_conv3(out["e2"]), leaky_alpha)
        out["e4"] = F.leaky_relu(self.enc_conv4(out["e3"]), leaky_alpha)
        out["e5"] = F.leaky_relu(self.enc_conv5(out["e4"]), leaky_alpha)
        out["e6"] = F.leaky_relu(self.enc_conv6(out["e5"]), leaky_alpha)
        out["e7"] = F.leaky_relu(self.enc_conv7(out["e6"]), leaky_alpha)
        out["e8"] = self.enc_conv8(out["e7"])

        out["d8"] = F.relu(F.dropout2d(self.dec_conv8(out["e8"]), training=True))
        out["d7"] = F.relu(F.dropout2d(self.dec_conv7(torch.cat((out["d8"], out["e7"]), dim=1)), training=True))
        out["d6"] = F.relu(F.dropout2d(self.dec_conv6(torch.cat((out["d7"], out["e6"]), dim=1)), training=True))
        out["d5"] = self.dec_conv5(torch.cat((out["d6"], out["e5"]), dim=1))
        out["d4"] = self.dec_conv4(torch.cat((out["d5"], out["e4"]), dim=1))
        out["d3"] = self.dec_conv3(torch.cat((out["d4"], out["e3"]), dim=1))
        out["d2"] = self.dec_conv2(torch.cat((out["d3"], out["e2"]), dim=1))
        out["d1"] = torch.tanh(self.dec_conv1(torch.cat((out["d2"], out["e1"]), dim=1)))

        return out["d1"]


class ResNetGenerator(nn.Module):
    """ Define a generator architecture based on the residual network for CycleGAN. """
    def __init__(self, in_channels, out_channels, num_features=64,
                 norm_layer=nn.BatchNorm2d, num_resblocks=6, padding_type="reflect"):
        """ Initialize the ResNet generator for CycleGAN. """
        super(ResNetGenerator, self).__init__()

        # First convolution layer
        model = [ConvBlock(in_channels, num_features,
                           kernel_size=7, padding_type="reflect", padding=3,
                           norm_layer=norm_layer)]

        # Downsampling layer
        num_downsampling = 2
        for i in range(num_downsampling):
            mul = 2 ** i
            model += [ConvBlock(num_features * mul, num_features * mul * 2,
                                kernel_size=3, padding_type="reflect", padding=1,
                                stride=2, norm_layer=norm_layer)]

        # Residual network layer
        mul = 2 ** num_downsampling
        for i in range(num_resblocks):
            model += [ResnetBlock(num_features * mul, padding_type=padding_type, norm_layer=norm_layer)]

        # Upsampling layer
        for i in range(num_downsampling):
            mul = 2 ** (num_downsampling - i)
            model += [DeconvBlock(num_features * mul, int(num_features * mul / 2),
                                  kernel_size=3, stride=2, padding=1, output_padding=1, norm_layer=norm_layer)]

        # Output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(num_features, out_channels, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        """ Forward function """
        return self.model(x)


class PatchDiscriminator(nn.Module):
    """ Define a PatchGAN discriminator. """
    def __init__(self, in_channels, num_features=64, norm_layer=nn.BatchNorm2d, leaky_alpha=0.2):
        """ Initialize the PatchGAN discriminator for CycleGAN. """
        super(PatchDiscriminator, self).__init__()

        # PatchGAN discriminator
        model = [nn.Conv2d(in_channels, num_features, kernel_size=4, stride=2),
                 nn.LeakyReLU(leaky_alpha, inplace=True)]

        for i in range(2):
            mul = 2 ** i
            model += [ConvBlock(num_features * mul, num_features * mul * 2,
                                kernel_size=4, padding=0, stride=2, norm_layer=norm_layer,
                                activation=None),
                      nn.LeakyReLU(leaky_alpha, inplace=True)]

        model += [ConvBlock(num_features * 4, num_features * 8,
                            kernel_size=4, padding=0, norm_layer=norm_layer,
                            activation=None),
                  nn.LeakyReLU(leaky_alpha, inplace=True)]

        model += [nn.Conv2d(num_features * 8, 1, kernel_size=4, stride=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        """ Forward function """
        return self.model(x)
