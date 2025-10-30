#!/usr/bin/env python3
"""train.py
An implementation of training the 3D Improved UNet3D module with code modified
from and based on [4].

Reference for [4] can be found in README.md
"""
import time
from matplotlib import pyplot as plt
from torch.utils.data import random_split, DataLoader
import dataset as ds
import modules as md
import torch
import torch.nn as nn
import random
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


class Trainer:
    """Handles the training of the model."""

    def __init__(self):
        """Initializes the training class."""
        # Paths
        self.rangpur_scan_path = \
            "/home/groups/comp3710/HipMRI_Study_open/semantic_MRs"
        self.rangpur_label_path = \
            "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only"
        self.pc_scan_path = "./semantic_MRs_anon"
        self.pc_label_path = "./semantic_labels_anon"
        self.model_save_path = "./output/model"
        self.image_save_path = "./output/results"

        # Hyper Parameters
        self.n_labels = 6
        self.batch_size = 2
        self.num_epochs = 300
        self.on_cluster = False
        self.learning_rate = 5e-4
        self.lr_schedule = 0.985
        self.weight_decay = 1e-5
        self.scale_data_size = 0.1
        self.validate_split = 0.1
        self.test_split = 0.1
        self.num_workers = 0

        # Components
        self.train_set = None
        self.test_set = None
        self.validate_set = None
        self.unused_set = None
        self.train_loader = None
        self.test_loader = None
        self.validate_loader = None
        self.unused_loader = None
        self.init_data_loaders()
        self.model = md.ImprovedUNet3D(1, self.n_labels, base_channels=8)

        # Device Config
        self.device = torch.device('cuda' if torch.cuda.is_available() else
                                   'cpu')
        print(f'Using device: {self.device}')

    def init_data_loaders(self):
        """Initialize the data loaders"""
        print("defining data loaders...")
        # pick data location and define dataset
        if self.on_cluster:
            dataset = ds.ProMRIDataSet(self.rangpur_scan_path,
                                       self.rangpur_label_path)
        else:
            dataset = ds.ProMRIDataSet(self.pc_scan_path, self.pc_label_path)
        # determine dataset sizes
        dataset_length = int(len(dataset) * self.scale_data_size)
        validate_size = int(self.validate_split * dataset_length)
        test_size = int(self.test_split * dataset_length)
        train_size = dataset_length - validate_size - test_size
        unused_size = len(dataset) - train_size - validate_size - test_size
        data_lengths = [train_size, validate_size, test_size, unused_size]
        # Distribute dataset among loaders
        self.train_set, self.validate_set, self.test_set, self.unused_set = random_split(dataset, data_lengths)
        self.train_loader = DataLoader(self.train_set)
        self.test_loader = DataLoader(self.test_set)
        self.validate_loader = DataLoader(self.validate_set)
        self.unused_loader = DataLoader(self.unused_set)
        print(
            f"data loaders created (using "
            f"{sum(data_lengths[:2])}/{len(dataset)}, "
            f"{(sum(data_lengths[:2]) / len(dataset)) * 100:.4f}%):\n"
            f"    train loader: size={len(self.train_loader)}\n"
            f"    validate loader: size={len(self.validate_loader)}\n"
            f"    test loader: size={len(self.test_loader)}\n"
            f"    unsued loader: size={len(self.unused_loader)}"
        )

    def train(self):
        """Train the model"""
        print("Training 3D Improved UNet3D...")
        self.model.to(self.device)
        criterion = MulticlassDiceLoss(self.n_labels)
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.learning_rate)

        start_time = time.time()
        next_vis = 1
        checkin_interval = 30
        checkin_time = time.time() + checkin_interval
        train_losses = []
        validate_losses = []

        print("Starting training...")
        for epoch in range(self.num_epochs):
            print(f'Epoch [{epoch + 1}/{self.num_epochs}]:')
            # Training loop with progress
            self.model.train()
            train_loss = 0
            for batch_idx, (images, masks) in enumerate(self.train_loader):
                images, masks = images.to(self.device), masks.to(self.device)

                # Forwards pass
                optimizer.zero_grad()
                outputs = self.model(images)

                # Calculate loss
                loss = criterion(outputs, masks)

                # Backward pass
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                # Check in
                if time.time() >= checkin_time:
                    print(f"    Checkin: time passed: "
                          f"{time.time() - start_time:.4f}, batch_inx: {batch_idx},"
                          f" curr_loss: {train_loss:.4f}")
                    checkin_time = time.time() + checkin_interval
            # update training losses
            avg_loss = train_loss / len(self.train_loader)
            train_losses.append(avg_loss)
            print(f'    Train Loss: {avg_loss:.4f}')

            # Validation Loop
            self.model.eval()
            validate_loss = 0
            with torch.no_grad():
                for batch_idx, (images, masks) in enumerate(self.validate_loader):
                    images, masks = images.to(self.device), masks.to(
                        self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, masks)
                    validate_loss += loss.item()
                    # Check in
                    if time.time() >= checkin_time:
                        print(f"    Checkin: time passed: "
                              f"{time.time() - start_time:.4f}, batch_inx: {batch_idx},"
                              f" curr_loss: {train_loss:.4f}")
                        checkin_time = time.time() + checkin_interval
            # update validation losses
            avg_loss = validate_loss / len(self.validate_loader)
            validate_losses.append(avg_loss)
            print(f'    Validate Loss: {avg_loss:.4f}')
            print(f"    Time: {(time.time() - start_time) / 60:.4f} min")

            # Visualize
            if ((epoch + 1) >= next_vis or epoch == 0 or epoch ==
                    self.num_epochs):
                vis_start_time = time.time()
                self.show_predictions(epoch)
                next_vis *= 2
                print(
                    f"    Visualization Time: "
                    f"{time.time() - vis_start_time:.4f} seconds")

        end = time.time()
        elapsed = end - start_time
        print(f"Training time: {elapsed:.4f} seconds / {elapsed/60:.4f} minutes")

    def show_predictions(self, epoch: int, n: int = 2):
        """Show model predictions.

        Args:
            epoch: the epoch number this is being printed on
            n: the number of extra random images to display
        """
        print("Performing prediction visualization...")
        self.model.eval()
        fig, axes = plt.subplots(3, n+1, figsize=(12, 9))
        fig.suptitle(f'Predictions After Epoch {epoch}', fontsize=16,
                     fontweight='bold')

        # get random index
        indexes = [0]
        for i in range(n):
            indexes.append(random.randint(0, len(self.validate_loader) - 1))

        with torch.no_grad():
            for plot_index, data_index in enumerate(indexes):
                images, true_masks = self.validate_set[data_index]
                images, true_masks = images.unsqueeze(0).to(self.device), true_masks.unsqueeze(0).to(self.device)

                outputs = self.model(images)
                # get the most likely label for each voxel
                # turns shape from [B, L, H, W, D] to [B, H, W, D]
                prediction_labels = torch.argmax(outputs, dim=1).cpu().numpy()
                true_labels = torch.argmax(true_masks, dim=1).cpu().numpy()

                # grab middle slices
                slice_idx = images.shape[2] // 2
                image = images[0, 0, slice_idx, :, :].cpu().numpy()
                prediction_label = prediction_labels[0, slice_idx, :, :]
                true_label = true_labels[0, slice_idx, :, :]

                # Plotting
                # Original Image
                axes[0, plot_index].imshow(image, cmap='grey')
                axes[0, plot_index].set_title(f'Original {data_index}', fontweight='bold')
                axes[0, plot_index].axis('off')
                # True Labels
                axes[1, plot_index].imshow(true_label, cmap='tab10', vmin=0, vmax=5)
                axes[1, plot_index].set_title(f'Ground Truth {data_index}', fontweight='bold')
                axes[1, plot_index].axis('off')
                # Prediction Labels
                axes[2, plot_index].imshow(prediction_label, cmap='tab10', vmin=0, vmax=5)
                accuracy = np.mean(prediction_label == true_label)
                axes[2, plot_index].set_title(
                    f'Prediction {data_index} (Acc: {accuracy:.3f})',
                    fontweight='bold')
                axes[2, plot_index].axis('off')

        plt.tight_layout()
        plt.show()


class MulticlassDiceLoss(nn.Module):
    """Dice Loss for multiclass segmentation. code modified to a Multiclass dice
    loss from [4].

    Dice Loss = 1 - Dice Coefficient.
    Dice Coefficient = (2/|K|) * sum((sum(u, v) / (sum(u) + sum(v))).

    Attributes:
        n_labels: The number of labels to calculate over.
        smooth: Smoothing factor to avoid division by zero (default: 1e-6).
    """

    def __init__(self, n_labels: int, smooth: float = 1e-6):
        """Initializes the multiclass dice loss class.

        Args:
            n_labels: The number of labels to calculate over.
            smooth: Smoothing factor to avoid division by zero (default: 1e-6).
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

trainer = Trainer()
trainer.train()
