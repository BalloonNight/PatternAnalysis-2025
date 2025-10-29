import dataset
import modules
import torch
import torch.nn as nn


# Paths
RANGPUR_SCAN_PATH = "/home/groups/comp3710/HipMRI_Study_open/semantic_MRs"
RANGPUR_LABEL_PATH = \
    "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only"
PC_SCAN_PATH = "/semantic_MRs_anon"
PC_LABEL_PATH = "/semantic_labels_anon"

# Training Parameters
N_LABELS = 6


class MulticlassDiceLoss(nn.Module):
    """Dice Loss for multiclass segmentation. code modified to a Multiclass dice
    loss from [4].

    Dice Loss = 1 - Dice Coefficient.
    Dice Coefficient = (2/|K|) * sum((sum(u, v) / (sum(u) + sum(v))).

    Attributes:
        smooth: Smoothing factor to avoid division by zero (default: 1e-6).
        n_labels: The number of labels to calculate over.
    """

    def __init__(self, smooth: float = 1e-6, n_labels: int = N_LABELS):
        """Initializes the multiclass dice loss class.

        Args:
            smooth: Smoothing factor to avoid division by zero (default: 1e-6).
            n_labels: The number of labels to calculate over.
        """
        super().__init__()
        self.smooth = smooth
        self.n_labels = n_labels

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Calculates the Multiclass Dice Loss of the prediction, tensors must
        be in the shape [B, L, H, W, D].

        Args:
            predictions: Output from model, shape: [B, L, H, W, D]
            targets: Ground truth, shape: [B, L, H, W, D]

        Returns:
            torch.Tensor: Multiclass Dice Loss
        """
        dice_coefficient = 0
        # go through each label type
        for l in range(self.n_labels):
            # Grab this label and flatten tensors
            prediction = predictions[:, l, :, :, :].reshape(-1)
            target = targets[:, l, :, :, :].reshape(-1).float()
            # Calculate intersection and union
            intersection = (prediction * target).sum()
            union = prediction.sum() + target.sum()
            # update dice coefficient
            dice_coefficient += ((2. * intersection + self.smooth) /
                                 (union + self.smooth))

        # Return Dice Loss (1 - Dice Coefficient)
        return 1 - (dice_coefficient / self.n_labels)
