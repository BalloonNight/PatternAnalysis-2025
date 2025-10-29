import dataset
import modules
import torch
import torch.nn as nn

# Rangpur MIR scan path: "/home/groups/comp3710/HipMRI_Study_open/semantic_MRs"
# Rangpur MIR label path: "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only"
# My computer MIR scan path: "/semantic_MRs_anon"
# My computer MIR label path: "/semantic_labels_anon"

# dice loss
# trainer
# visualiser

class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation. code from [4]

    Dice Loss = 1 - Dice Coefficient
    Dice Coefficient = (2 * |X ∩ Y|) / (|X| + |Y|)

    Args:
        smooth (float): Smoothing factor to avoid division by zero (default: 1e-6)
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Sigmoid output from model [B, H, W] (values between 0-1)
            targets: Binary ground truth [B, H, W] (values 0 or 1)
        """
        # Flatten tensors using reshape to handle non-contiguous memory layout
        predictions = predictions.reshape(-1)
        targets = targets.reshape(-1).float()

        # Calculate intersection and union
        intersection = (predictions * targets).sum()
        dice_coeff = (2.0 * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)

        # Return Dice Loss (1 - Dice Coefficient)
        return 1 - dice_coeff
