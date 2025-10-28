#!/usr/bin/env python3
"""
model.py
An implementation of the 3D Improved UNet3D module from [2].

References for [2] can be found in README.md
"""
import torch
import torch.nn as nn

DEFAULT_DROPOUT = 0.3
NEGATIVE_SLOPE = 1e-2
N_CLASSES = 6
BASE_FILTERS = 16

class ContextModule(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p=DEFAULT_DROPOUT):
        super().__init__()
        self.norm1 = nn.InstanceNorm3d(in_ch)
        self.relu1 = nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE)
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.drop1 = nn.Dropout3d(p=dropout_p)
        self.norm2 = nn.InstanceNorm3d(out_ch)
        self.relu2 = nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.norm1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.drop1(out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        return out


class LocalizationModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.norm1 = nn.InstanceNorm3d(in_ch)
        self.relu1 = nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE)
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_ch)
        self.relu2 = nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=1, stride=1,
                               padding=0, bias=False)

    def forward(self, x):
        out = self.norm1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        return out


class UpsamplingModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upsp1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.upsp1(x)
        out = self.conv1(out)
        return out

class ImprovedUNet3D2(nn.Module):
    def __init__(self, in_ch: int, filters=BASE_FILTERS, n_classes=N_CLASSES,
                 dropout_p=DEFAULT_DROPOUT):
        super().__init__()
        self.in_ch = in_ch
        self.filters = filters
        self.n_classes = n_classes
        self.dropout_p = dropout_p

        # Encoder
        self.conv1 = nn.Conv3d(self.in_ch, self.filters, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.cont1 = ContextModule(self.filters, self.filters, self.dropout_p)
        self.conv2 = nn.Conv3d(self.filters, self.filters * 2, kernel_size=3,
                               stride=2, padding=1, bias=False)
        self.cont2 = ContextModule(self.filters * 2, self.filters * 2,
                                   self.dropout_p)
        self.conv3 = nn.Conv3d(self.filters * 2, self.filters * 4,
                               kernel_size=3, stride=2, padding=1, bias=False)
        self.cont3 = ContextModule(self.filters * 4, self.filters * 4,
                                   self.dropout_p)
        self.conv4 = nn.Conv3d(self.filters * 4, self.filters * 8,
                               kernel_size=3, stride=2, padding=1, bias=False)
        self.cont4 = ContextModule(self.filters * 8, self.filters * 8,
                                   self.dropout_p)
        self.conv5 = nn.Conv3d(self.filters * 8, self.filters * 16,
                               kernel_size=3, stride=2, padding=1, bias=False)
        self.cont5 = ContextModule(self.filters * 16, self.filters * 16,
                                   self.dropout_p)

        # Decoder
        self.upsp1 = UpsamplingModule(self.filters * 16, self.filters * 8)
        self.loca1 = LocalizationModule(self.filters * 8, self.filters * 8)
        self.upsp2 = UpsamplingModule(self.filters * 8, self.filters * 4)
        self.loca2 = LocalizationModule(self.filters * 4, self.filters * 4)
        self.upsp3 = UpsamplingModule(self.filters * 4, self.filters * 2)
        self.loca3 = LocalizationModule(self.filters * 2, self.filters * 2)
        self.upsp4 = UpsamplingModule(self.filters * 2, self.filters)
        self.conv6 = nn.Conv3d(self.filters, self.filters * 2, kernel_size=3,
                               stride=1, padding=1, bias=False)

        # Segmentation
        self.segm1 = nn.Conv3d(self.filters * 4, self.n_classes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.upsc1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.segm2 = nn.Conv3d(self.filters * 2, self.n_classes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.upsc2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.segm3 = nn.Conv3d(self.filters, self.n_classes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.smax = nn.Softmax(dim=1)

    def forward(self, x):
        # Encoder
        out = self.conv1(x)
        res = out
        out = self.cont1(out)
        out += res
        residual1 = out
        out = self.conv2(out)
        res = out
        out = self.cont2(out)
        out += res
        residual2 = out
        out = self.conv3(out)
        res = out
        out = self.cont3(out)
        out += res
        residual3 = out
        out = self.conv4(out)
        res = out
        out = self.cont4(out)
        out += res
        residual4 = out
        out = self.conv5(out)
        res = out
        out = self.cont5(out)
        out += res

        # Decoder
        out = self.upsp1(out)
        out = torch.cat([out, residual4], dim=1)
        out = self.loca1(out)
        out = self.upsp2(out)
        out = torch.cat([out, residual3], dim=1)
        out = self.loca2(out)
        to_seg1 = out
        out = self.upsp3(out)
        out = torch.cat([out, residual2], dim=1)
        out = self.loca3(out)
        to_seg2 = out
        out = self.upsp4(out)
        out = torch.cat([out, residual1], dim=1)
        out = self.conv6(out)

        # Segmentation
        out_temp1 = self.segm1(to_seg1)
        out_temp1 = self.upsc1(out_temp1)
        out_temp2 = self.segm2(to_seg2)
        out_temp2 += out_temp1
        out_temp2 = self.upsc2(out_temp2)
        out = self.segm3(out)
        out += out_temp2
        out = self.smax(out)

        return out
