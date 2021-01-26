
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_networks import *


class UNet(nn.Module):
    """ U-Net architecture """
    def __init__(self, in_channels, out_channels, num_features=32,
                 feature_mode="fixed", pool="avg", upsample_mode="bilinear"):
        """ Initialize the U-Net architecture
        Parameters:
            in_channels (int): the number of input channels
            out_channels (int): the number of output channels
            num_features (int): the number of features in the first layer
            feature_mode (str): feature-increasing mode along the depth ("fixed" or "pyramid")
            pool (str): pooling method ("max" or "avg")
            upsample_mode (str): upsampling method ("bilinear" or "nearest")
        """
        super(UNet, self).__init__()

        # Assert
        assert feature_mode in ["fixed", "pyramid"]
        assert pool in ["max", "avg"]
        assert upsample_mode in ["nearest", "bilinear"]

        # Layer definition
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) if pool == "max" \
            else nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.enc_conv0 = ConvBlock2(in_channels=in_channels, out_channels=num_features)

        if feature_mode == "fixed":
            self.enc_conv1 = ConvBlock2(in_channels=num_features, out_channels=num_features)
            self.enc_conv2 = ConvBlock2(in_channels=num_features, out_channels=num_features)
            self.enc_conv3 = ConvBlock2(in_channels=num_features, out_channels=num_features)
            self.enc_conv4 = ConvBlock2(in_channels=num_features, out_channels=num_features)

            self.up_conv4 = UpConv(in_channels=num_features, out_channels=num_features, mode=upsample_mode)
            self.up_conv3 = UpConv(in_channels=num_features, out_channels=num_features, mode=upsample_mode)
            self.up_conv2 = UpConv(in_channels=num_features, out_channels=num_features, mode=upsample_mode)
            self.up_conv1 = UpConv(in_channels=num_features, out_channels=num_features, mode=upsample_mode)

            self.dec_conv4 = ConvBlock2(in_channels=2 * num_features, out_channels=num_features)
            self.dec_conv3 = ConvBlock2(in_channels=2 * num_features, out_channels=num_features)
            self.dec_conv2 = ConvBlock2(in_channels=2 * num_features, out_channels=num_features)
            self.dec_conv1 = ConvBlock2(in_channels=2 * num_features, out_channels=num_features)
        else:
            self.enc_conv1 = ConvBlock2(in_channels=1 * num_features, out_channels=2 * num_features)
            self.enc_conv2 = ConvBlock2(in_channels=2 * num_features, out_channels=4 * num_features)
            self.enc_conv3 = ConvBlock2(in_channels=4 * num_features, out_channels=8 * num_features)
            self.enc_conv4 = ConvBlock2(in_channels=8 * num_features, out_channels=16 * num_features)

            self.up_conv4 = UpConv(in_channels=16 * num_features, out_channels=8 * num_features, mode=upsample_mode)
            self.up_conv3 = UpConv(in_channels=8 * num_features, out_channels=4 * num_features, mode=upsample_mode)
            self.up_conv2 = UpConv(in_channels=4 * num_features, out_channels=2 * num_features, mode=upsample_mode)
            self.up_conv1 = UpConv(in_channels=2 * num_features, out_channels=1 * num_features, mode=upsample_mode)

            self.dec_conv4 = ConvBlock2(in_channels=16 * num_features, out_channels=8 * num_features)
            self.dec_conv3 = ConvBlock2(in_channels=8 * num_features, out_channels=4 * num_features)
            self.dec_conv2 = ConvBlock2(in_channels=4 * num_features, out_channels=2 * num_features)
            self.dec_conv1 = ConvBlock2(in_channels=2 * num_features, out_channels=1 * num_features)

        self.conv_1x1 = nn.Conv2d(in_channels=num_features, out_channels=out_channels, kernel_size=(1, 1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Encoding path
        enc0 = self.enc_conv0(x)
        enc1 = self.enc_conv1(self.pool(enc0))
        enc2 = self.enc_conv2(self.pool(enc1))
        enc3 = self.enc_conv3(self.pool(enc2))
        enc4 = self.enc_conv4(self.pool(enc3))

        # Decoding path with skip connection
        dec = self.dec_conv4(torch.cat((self.up_conv4(enc4), enc3), dim=1))
        dec = self.dec_conv3(torch.cat((self.up_conv3(dec), enc2), dim=1))
        dec = self.dec_conv2(torch.cat((self.up_conv2(dec), enc1), dim=1))
        dec = self.dec_conv1(torch.cat((self.up_conv1(dec), enc0), dim=1))

        # 1x1 Conv
        y = self.conv_1x1(dec)
        y = self.softmax(y)

        return y