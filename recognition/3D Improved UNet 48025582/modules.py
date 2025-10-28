import torch
import torch.nn as nn

DEFAULT_DROPOUT = 0.3
NEGATIVE_SLOPE = 10 ^ -2

class UNet3D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p=DEFAULT_DROPOUT):
        super().__init__()
        # Network architecture blocks, times used:
            # 3x3x3 Convolution, 2
            # context module, 5, [13]
                # pre-activation residual block
                # two 3x3x3 convolutional layers
                # dropout layer with pdrop=0.3 inbetween
            # 3x3x3 stride 2 convolution, 4
            # upsampling module, 4
            # localization module, 3
                # upsample low resolution feature maps
                    # simple upscale that repeats the feature voxels twice in each spatial dimension
            # segmentation layer, 3
            # softmax, 1
            # element wise sum, 7, just + them
            # concatenation, 4, torch.cat
            # upscale 2
        # use leaky ReLU
        # negative slope 10^-2
        # replace traditional batch with instance normalisation [14]

        # Encoder

    def _context_module(self, in_ch, out_ch, dropout_p=DEFAULT_DROPOUT):
        pass

    def _localization_module(self, in_ch, out_ch, dropout_p=DEFAULT_DROPOUT):
        pass

    def _upsample_module(self, in_ch, out_ch, dropout_p=DEFAULT_DROPOUT):
        pass

    def forward(self, x):
        return x

class DiceLoss(nn.Module):
    pass
