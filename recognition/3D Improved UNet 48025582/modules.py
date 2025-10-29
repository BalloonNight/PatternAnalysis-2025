#!/usr/bin/env python3
"""model.py
An implementation of the 3D Improved UNet3D module from [2]. With code loosely
inspired for pytorch functions and some structures [3].

References for [2] and [3] can be found in README.md
"""
import torch
import torch.nn as nn

# global constants
DEFAULT_DROPOUT = 0.3
NEGATIVE_SLOPE = 1e-2
N_CLASSES = 6
BASE_FILTERS = 16

class ContextModule(nn.Module):
    """This class handles the context module for the UNet algorithm.

    as per [2], this is a pre-activation residual block with 2 3x3x3 convolution
    layers and a dropout layer in between and a skip connection from the input
    to output. does include the conv always before the context model too and
    implements the skip afterwards like normal.

    Attributes:
        convolution1: pre conv layer
        normalize1: 1st layer InstanceNorm3d.
        relu1: 1st layer LeakyReLU.
        convolution2: 1st layer Conv3d.
        dropout: Dropout3d layer.
        normalize2: 2nd layer InstanceNorm3d.
        relu2: 2nd layer LeakyReLU.
        convolution3: 2nd layer Conv3d.
    """

    def __init__(self, in_channels: int, out_channels: int = None, stride:
    int = 2, dropout_p: float =
    DEFAULT_DROPOUT):
        """Initializes the Context Module.

        Args:
            in_channels (int): num of in channels.
            out_channels (int): num of out channels, default double in channels.
            stride (int): stride, default 2.
            dropout_p (float): probability of each channel being randomly zeroed
             out, default DEFAULT_DROPOUT.
        """
        super().__init__()
        if out_channels is None:
            out_channels = in_channels * 2
        # pre conv
        self.convolution1 = nn.Conv3d(in_channels, out_channels, kernel_size=3,
                                      stride=stride, padding=1, bias=False)
        # layer 1
        self.normalize1 = nn.InstanceNorm3d(out_channels)
        self.relu1 = nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE)
        self.convolution2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                                      stride=1, padding=1, bias=False)
        # connection
        self.dropout = nn.Dropout3d(p=dropout_p)
        # layer 2
        self.normalize2 = nn.InstanceNorm3d(out_channels)
        self.relu2 = nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE)
        self.convolution3 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                                      stride=1, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the context module.

        Args:
            x (torch.Tensor): The input feature map.

        Returns:
            torch.Tensor: the forward passed feature map through the context
             module.
        """
        # pre conv
        out = self.convolution1(x)
        residual = out
        # layer 1
        out = self.normalize1(out)
        out = self.relu1(out)
        out = self.convolution2(out)
        # connection
        out = self.dropout(out)
        # layer 2
        out = self.normalize2(out)
        out = self.relu2(out)
        out = self.convolution3(out)
        # add skip connection
        return out + residual


class LocalizationModule(nn.Module):
    """This class handles the localization module for the algorithm.

    as per [2], this consists of a 3x3x3 convolution followed by a 1x1x1
    convolution which halves the number of feature maps.

    Attributes:
        normalization1: 1st layer InstanceNorm3d.
        relu1: 1st layer LeakyReLU.
        convolution1: 1st layer Conv3d.
        normalization2: 2nd layer InstanceNorm3d.
        relu2: 2nd layer LeakyReLU.
        convolution2: 2nd layer Conv3d.
    """

    def __init__(self, in_channels: int):
        """Initializes the Localization Module.

        Args:
            in_channels: number of input channels
        """
        super().__init__()
        out_channels = in_channels // 2
        # layer 1
        self.normalization1 = nn.InstanceNorm3d(in_channels)
        self.relu1 = nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE)
        self.convolution1 = nn.Conv3d(in_channels, in_channels, kernel_size=3,
                                      stride=1, padding=1, bias=False)
        # layer 2
        self.normalization2 = nn.InstanceNorm3d(in_channels)
        self.relu2 = nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE)
        self.convolution2 = nn.Conv3d(in_channels, out_channels, kernel_size=1,
                                      stride=1, padding=0, bias=False)

    def forward(self, x: torch.Tensor):
        """Performs a forward pass through the localization module.

        Args:
            x (torch.Tensor): The input feature map.

        Returns:
            torch.Tensor: the forward passed feature map through the
             localization module.
        """
        # layer 1
        out = self.normalization1(x)
        out = self.relu1(out)
        out = self.convolution1(out)
        # layer 2
        out = self.normalization2(out)
        out = self.relu2(out)
        out = self.convolution2(out)
        return out


class UpsamplingModule(nn.Module):
    """This class handles the upsampling module for the algorithm.

    As per [2], this consists an upscale that repeats feature voxels twice, and
    then a 3x3x3 convolution that half's the feature map.

    Attributes:
        upsample: A Upsample layer.
        convolution: A Conv3d layer.
    """

    def __init__(self, in_channels):
        """Initializes the Upsampling Module.

        Args:
            in_channels: number of input channels
        """
        super().__init__()
        out_channels = in_channels // 2
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.convolution = nn.Conv3d(in_channels, out_channels, kernel_size=3,
                                     stride=1, padding=1, bias=False)

    def forward(self, x: torch.Tensor):
        """Performs a forward pass through the upsampling module.

        Args:
            x (torch.Tensor): The input feature map.

        Returns:
            torch.Tensor: the forward passed feature map through the
             upsampling module.
        """
        out = self.upsample(x)
        out = self.convolution(out)
        return out

class ImprovedUNet3D(nn.Module):
    """This class handles the 3D Improved UNet3D model.

    as per [2], this consists of a 5 layer context pathway and 5 layer
    localization pathway, with residual connections between the pathways, and
    the last 3 layers of the localization pathway being combined for the final
    output. The number in the names of the modules in attributes represents
    their depth in the model, which is determined by the verticality of the
    module in diagram, figure [INSERT FIGURE NUMBER], in README.md.

    Attributes:
        in_channels: number of starting channels.
        filters: number of filters.
        n_classes: number of classes.
        dropout_p: chance of channels being set to 0 during training.
        context1: 1st layer context module.
        context2: 2nd layer context module.
        context3: 3rd layer context module.
        context4: 4th layer context module.
        context5: 5th layer context module.
        upsample5: 5th layer upsample module.
        localization4: 4th layer localization module.
        upsample4: 4th layer upsample module.
        localization3: 3rd layer localization module.
        upsample3: 3rd layer upsample module.
        localization2: 2nd layer localization module.
        upsample2: 2nd layer upsample module.
        convolution1: 1st layer localization module.
        segment3: 3rd layer segmentation layer.
        upscale3: 3rd layer upscale.
        segment2: 2nd layer segmentation layer.
        upscale2: 2nd layer upscale.
        segment1: 1st layer segmentation layer.
        smax1: 1st layer softmax.
    """
    def __init__(self, in_channels: int, filters=BASE_FILTERS,
                 n_classes=N_CLASSES, dropout_p=DEFAULT_DROPOUT):
        """Initializes the 3D Improved UNet 3D module.

        Args:
            in_channels: number of starting channels.
            filters: number of filters.
            n_classes: number of classes.
            dropout_p: chance of channels being set to 0 during training.
        """
        super().__init__()
        # input vars
        self.in_channels = in_channels
        self.filters = filters
        self.n_classes = n_classes
        self.dropout_p = dropout_p

        # Encoder/Context Pathway
        self.context1 = ContextModule(self.in_channels, self.filters, 1,
                                      self.dropout_p)
        self.context2 = ContextModule(self.filters, dropout_p=self.dropout_p)
        self.context3 = ContextModule(self.filters * 2,
                                      dropout_p=self.dropout_p)
        self.context4 = ContextModule(self.filters * 4,
                                      dropout_p=self.dropout_p)
        self.context5 = ContextModule(self.filters * 8,
                                      dropout_p=self.dropout_p)

        # Decoder/Localization Pathway
        self.upsample5 = UpsamplingModule(self.filters * 16)
        self.localization4 = LocalizationModule(self.filters * 16)
        self.upsample4 = UpsamplingModule(self.filters * 8)
        self.localization3 = LocalizationModule(self.filters * 8)
        self.upsample3 = UpsamplingModule(self.filters * 4)
        self.localization2 = LocalizationModule(self.filters * 4)
        self.upsample2 = UpsamplingModule(self.filters * 2)
        self.convolution1 = nn.Conv3d(
            self.filters * 2, self.filters * 2,
            kernel_size=3, stride=1, padding=1, bias=False)

        # Segmentation
        self.segment3 = nn.Conv3d(
            self.filters * 4, self.n_classes,
            kernel_size=1, stride=1, padding=0, bias=False)
        self.upscale3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.segment2 = nn.Conv3d(
            self.filters * 2, self.n_classes,
            kernel_size=1, stride=1, padding=0, bias=False)
        self.upscale2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.segment1 = nn.Conv3d(
            self.filters * 2, self.n_classes,
            kernel_size=1, stride=1, padding=0, bias=False)
        self.smax1 = nn.Softmax(dim=1)

    def forward(self, x):
        """Performs a forward pass through the 3D Improved UNet3D module.

        Args:
            x (torch.Tensor): The input feature map.

        Returns:
            torch.Tensor: the forward passed feature map through the
             3D Improved UNet3D module.
        """
        # Encoder/Context Pathway
        out = self.context1(x)
        context_layer_1_out = out
        out = self.context2(out)
        context_layer_2_out = out
        out = self.context3(out)
        context_layer_3_out = out
        out = self.context4(out)
        context_layer_4_out = out
        out = self.context5(out)

        # Decoder/Localization Pathway
        out = self.upsample5(out)
        # note: concatenation doubles the number of output filters
        out = torch.cat([out, context_layer_4_out], dim=1)
        out = self.localization4(out)
        out = self.upsample4(out)
        out = torch.cat([out, context_layer_3_out], dim=1)
        out = self.localization3(out)
        localization_layer_3_out = out
        out = self.upsample3(out)
        out = torch.cat([out, context_layer_2_out], dim=1)
        out = self.localization2(out)
        localization_layer_2_out = out
        out = self.upsample2(out)
        out = torch.cat([out, context_layer_1_out], dim=1)
        out = self.convolution1(out)

        # Segmentation
        layer_3_segment = self.segment3(localization_layer_3_out)
        layer_3_segment = self.upscale3(layer_3_segment)
        layer_2_segment = self.segment2(localization_layer_2_out)
        layer_2_segment += layer_3_segment
        layer_2_segment = self.upscale2(layer_2_segment)
        out = self.segment1(out)
        out += layer_2_segment
        out = self.smax1(out)

        return out
