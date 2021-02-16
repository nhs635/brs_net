
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
        self.sigmoid = nn.Sigmoid()

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
        y = self.sigmoid(y)

        return y


class BrsResSegNet(nn.Module):
    """ BRS segmentation and detection network """
    def __init__(self, in_channels, out_channels):
        """ Initialize the BRS seg net architecture
        Parameters:
            in_channels (int): the number of input channels
            out_channels (int): the number of output channels
        """
        super(BrsResSegNet, self).__init__()

        # Layer definition
        self.conv_block1 = ConvBlock(in_channels=in_channels, out_channels=32,
                                     kernel_size=(7, 7), padding=(3, 3, 3, 3), padding_type="reflect")

        self.res_encoder1 = ResidualBlock(in_channels=32, out_channels=64,
                                          padding_type="reflect", downsample=True)
        self.res_encoder2 = ResidualBlock(in_channels=64, out_channels=128,
                                          padding_type="reflect", downsample=True)
        self.res_encoder3 = ResidualBlock(in_channels=128, out_channels=256,
                                          padding_type="reflect", downsample=True)
        self.res_encoder4 = ResidualBlock(in_channels=256, out_channels=512,
                                          padding_type="reflect", downsample=True)

        self.dilated_module = DilatedModule(in_channels=512, intmed_channels=512)

        self.res_decoder4 = ResidualBlock(in_channels=512, out_channels=512, padding_type="reflect")
        self.subpix_conv4 = SubPixelConv(num_features=512)
        self.res_decoder3 = ResidualBlock(in_channels=768, out_channels=256, padding_type="reflect")
        self.subpix_conv3 = SubPixelConv(num_features=256)
        self.res_decoder2 = ResidualBlock(in_channels=384, out_channels=128, padding_type="reflect")
        self.subpix_conv2 = SubPixelConv(num_features=128)
        self.res_decoder1 = ResidualBlock(in_channels=192, out_channels=64, padding_type="reflect")
        self.subpix_conv1 = SubPixelConv(num_features=64)
        self.res_decoder0 = ResidualBlock(in_channels=96, out_channels=32, padding_type="reflect")
        self.subpix_conv0 = SubPixelConv(num_features=32)

        self.global_conv_module = GlobalConvModule(in_channels=32, out_channels=out_channels)

        self.dropout = nn.Dropout2d(p=0.5)
        self.g_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Linear(in_features=512, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=out_channels)

    def forward(self, x):
        x0 = self.dropout(self.conv_block1(x))
        x1 = self.dropout(self.res_encoder1(x0))
        x2 = self.dropout(self.res_encoder2(x1))
        x3 = self.dropout(self.res_encoder3(x2))
        x4 = self.dropout(self.res_encoder4(x3))
        dm = self.dilated_module(x4)

        gp = torch.squeeze(self.g_avg_pool(dm))
        t = torch.sigmoid(self.fc2(self.fc1(gp)))

        dm = self.dropout(dm)
        y4 = self.dropout(self.subpix_conv4(self.res_decoder4(dm)))
        y3 = self.dropout(self.subpix_conv3(self.res_decoder3(torch.cat((y4, x3), dim=1))))
        y2 = self.dropout(self.subpix_conv2(self.res_decoder2(torch.cat((y3, x2), dim=1))))
        y1 = self.dropout(self.subpix_conv1(self.res_decoder1(torch.cat((y2, x1), dim=1))))
        y0 = self.dropout(self.res_decoder0(torch.cat((y1, x0), dim=1)))

        y = torch.sigmoid(self.global_conv_module(y0))

        return y
