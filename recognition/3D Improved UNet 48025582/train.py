#!/usr/bin/env python3
"""train.py
An implementation of training the 3D Improved UNet3D module.

Reference for [4] and [5] can be found in README.md
"""
import os
import re
import shutil
import time
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from torch.utils.data import random_split, DataLoader
import dataset as ds
import modules as md
import torch
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

        # File saving
        self.training_time = int(time.time())
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.image_save_path, exist_ok=True)

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
        self.smooth = 1e-6

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
        self.train_set, self.validate_set, self.test_set, self.unused_set = (
            random_split(dataset, data_lengths))
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
        """Train the model, code modified from [4]"""
        print("Training 3D Improved UNet3D...")
        self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.learning_rate)

        # Tracking times
        start_time = time.time()
        next_vis = 1
        checkin_interval = 30
        checkin_time = time.time() + checkin_interval

        # Tracking Data
        train_losses = []
        train_coefficients = []
        train_accuracy = []
        validate_losses = []
        validate_coefficients = []
        validate_accuracy = []

        # start training loop
        print("Starting training...")
        for epoch in range(self.num_epochs):
            print(f'Epoch [{epoch + 1}/{self.num_epochs}]:')
            epoch_start_time = time.time()
            # Training loop with progress
            self.model.train()
            multi_dice_loss = 0
            dice_coefficients = [0, 0, 0, 0, 0, 0, 0]
            accuracy = 0
            for batch_idx, (images, true_labels) in enumerate(
                    self.train_loader):
                images, true_labels = images.to(self.device), true_labels.to(
                    self.device)

                # Forwards pass
                optimizer.zero_grad()
                predict_labels = self.model(images)

                # Calculate loss
                cur_loss, cur_coefficients = self.multiclass_dice_loss(
                    predict_labels, true_labels)
                accuracy += self.compute_accuracy(predict_labels, true_labels)

                # Backward pass
                cur_loss.backward()
                optimizer.step()

                # update current epoches loss's
                multi_dice_loss += cur_loss.item()
                dice_coefficients = [x + y for x, y in zip(dice_coefficients,
                                                           cur_coefficients)]

                # Check in
                if time.time() >= checkin_time:
                    print(f"    Checkin:\n"
                          f"        time passed: {time.time() - start_time:.4f}\n"
                          f"        batch_inx: {batch_idx}\n"
                          f"        curr_loss: {multi_dice_loss:.4f}\n"
                          f"        curr_coefficient: {[f'{c:.4f}' for c in dice_coefficients]}\n"
                          f"        accuracy: {accuracy}")
                    checkin_time = time.time() + checkin_interval

            # update training losses
            avg_loss = multi_dice_loss / len(self.train_loader)
            avg_coefficients = [x / len(self.train_loader) for x in
                                dice_coefficients]
            avg_accuracy = accuracy / len(self.train_loader)
            train_losses.append(avg_loss)
            train_coefficients.append(avg_coefficients)
            train_accuracy.append(avg_accuracy)
            print(f"    Train:\n"
                  f"        Loss: {avg_loss:.4f}\n"
                  f"        Coefficients: {[f'{c:.4f}' for c in dice_coefficients]}\n"
                  f"        Accuracy: {avg_accuracy}")

            # Validation Loop
            self.model.eval()
            multi_dice_loss = 0
            dice_coefficients = [0, 0, 0, 0, 0, 0, 0]
            accuracy = 0
            with torch.no_grad():
                for batch_idx, (images, true_labels) in enumerate(
                        self.validate_loader):
                    images, true_labels = (images.to(self.device),
                                           true_labels.to(self.device))
                    predict_labels = self.model(images)
                    cur_loss, cur_coefficients = self.multiclass_dice_loss(
                        predict_labels, true_labels)
                    multi_dice_loss += cur_loss.item()
                    dice_coefficients = [x + y for x, y in zip(
                        dice_coefficients, cur_coefficients)]
                    accuracy += self.compute_accuracy(predict_labels,
                                                      true_labels)

                    # Check in
                    if time.time() >= checkin_time:
                        print(f"    Checkin:\n"
                              f"        time passed: {time.time() - start_time:.4f}\n"
                              f"        batch_inx: {batch_idx}\n"
                              f"        curr_loss: {multi_dice_loss:.4f}\n"
                              f"        curr_coefficient: {[f'{c:.4f}' for c in dice_coefficients]}\n"
                              f"        accuracy: {accuracy}")
                        checkin_time = time.time() + checkin_interval

            # update training losses
            avg_loss = multi_dice_loss / len(self.validate_loader)
            avg_coefficients = [x / len(self.validate_loader) for x in
                                dice_coefficients]
            avg_accuracy = accuracy / len(self.validate_loader)
            validate_losses.append(avg_loss)
            validate_coefficients.append(avg_coefficients)
            validate_accuracy.append(avg_accuracy)
            print(f"    Validation:\n"
                  f"        Loss: {avg_loss:.4f}\n"
                  f"        Coefficients: {[f'{c:.4f}' for c in dice_coefficients]}\n"
                  f"        Accuracy: {avg_accuracy}")

            # Visualize
            if (epoch >= next_vis or epoch == 0 or epoch ==
                    self.num_epochs):
                vis_start_time = time.time()
                self.show_predictions(epoch)
                next_vis *= 2
                print(
                    f"    Visualization Time: "
                    f"{time.time() - vis_start_time:.4f} seconds")

            epoch_time = time.time() - epoch_start_time
            print(f"    Epoch time: {epoch_time}")

        end = time.time()
        elapsed = end - start_time
        print(f"Training time: {elapsed:.4f} seconds / "
              f"{elapsed/60:.4f} minutes")

    def show_predictions(self, epoch: int):
        """Show model predictions. Code modified from [4].

        Args:
            epoch: the epoch number this is being printed on.
        """
        print("Performing prediction visualization...")
        # get the data to display
        self.model.eval()
        with torch.no_grad():
            # get the first validate image and true label
            image, true_label = next(iter(self.validate_loader))

            # predict the label
            predict_label = self.model(image.to(self.device))

            # Reshape from [B, C, H, W, D] to [H, W, D]
            image = image[0, 0, :, :, :].cpu().numpy()
            true_label = torch.argmax(true_label, dim=1)[0, :, :, :
            ].cpu().numpy()
            predict_label = torch.argmax(predict_label, dim=1)[0, :, :, :
            ].cpu().numpy()

            # Plotting
            title = f"Epoch_{epoch}_Prediction"
            self.plot_3d_image(image, true_label, predict_label, title)

    def multiclass_dice_loss(self, predictions: torch.Tensor,
                             targets: torch.Tensor) \
            -> tuple[float, list[float]]:
        """Calculates the Multiclass Dice Loss of the prediction, tensors must
        be in the shape [B, L, H, W, D]. Code modified from [4].

        Dice Loss = 1 - Dice Coefficient.
        Dice Coefficient = (2/|K|) * sum((sum(u, v) / (sum(u) + sum(v))).

        Args:
            predictions: Output from model, shape: [B, L, H, W, D]
            targets: Ground truth, shape: [B, L, H, W, D]

        Returns:
            tuple[float, list[float]]: Multiclass Dice Loss, List of each
             individual labels and the multiclass Dice coefficient.
        """
        total_dice_coefficient = 0.0
        dice_coefficients = []
        # go through each label type
        for l in range(self.n_labels):
            # Grab this label and flatten tensors
            prediction = predictions[:, l, :, :, :].reshape(-1)
            target = targets[:, l, :, :, :].reshape(-1).float()
            # Calculate intersection and union
            intersection = (prediction * target).sum()
            union = prediction.sum() + target.sum()
            # update dice coefficient
            dice_coefficient = ((2.0 * intersection + self.smooth) /
                                (union + self.smooth))
            total_dice_coefficient += dice_coefficient
            dice_coefficients.append(dice_coefficient)

        avg_dice_coefficient = total_dice_coefficient / self.n_labels
        dice_coefficients.append(avg_dice_coefficient)

        # Return Dice Loss (1 - Dice Coefficient)
        return 1 - avg_dice_coefficient, dice_coefficients

    def plot_3d_image(self, image: np.ndarray, true_label: np.ndarray,
                      predict_label: np.ndarray, title: str):
        """Plots a 3d image by scrolling through one of its axis. modified from
        [5]. The images must be of shape [H, W, D].

        Args:
            image: The 3D image to be plotted.
            true_label: The true labeling for the MRI scan.
            predict_label: The predicted labeling for the MRI scan.
            title: The title to be put into the graph and file save name.
        """
        image = np.flip(np.transpose(image, axes=[0, 2, 1]), axis=1)
        true_label = np.flip(np.transpose(true_label, axes=[0, 2, 1]), axis=1)
        predict_label = np.flip(np.transpose(predict_label, axes=[0, 2, 1]),
                                axis=1)

        fig, axes = plt.subplots(3, figsize=(6, 12))
        axes[0].set_title(f"Image")
        axes[1].set_title(f"Real Label")
        axes[2].set_title(f"Predict Label")

        vmin_img = np.min(image)
        vmax_img = np.max(image)

        imshow1 = axes[0].imshow(image[0], cmap='gray', vmin=vmin_img,
                                 vmax=vmax_img, animated=True)
        imshow2 = axes[1].imshow(true_label[0], cmap='tab10', vmin=0, vmax=5,
                                 animated=True)
        imshow3 = axes[2].imshow(predict_label[0], cmap='tab10', vmin=0, vmax=5,
                                 animated=True)

        def update(frame):
            imshow1.set_array(image[frame])
            imshow2.set_array(true_label[frame])
            imshow3.set_array(predict_label[frame])
            return [imshow1, imshow2, imshow3]

        ani = animation.FuncAnimation(fig, update, frames=image.shape[0],
                                      interval=1, blit=True, repeat=True)

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()

        writer = animation.PillowWriter(fps=120, metadata=dict(artist='Me'),
                                        bitrate=1800)
        save_path = self.get_formatted_filepath(self.image_save_path, title, "gif")
        ani.save(save_path, writer=writer, dpi=80)
        return ani

    def get_formatted_filepath(self, path: str, name: str, type: str) -> str:
        """Given a filename returns a formatted one with its full path better for saving.

        Args:
            path: path the file will be going
            name: unformatted name of the file.
            type: the name of the filetype

        Returns:
            str: properly formatted filepath.
        """
        # Get rid of weird characters
        filename = "".join(char if char.isalnum() else "_" for char in name)
        filename = re.sub(r'_+', '_', filename)  # get rid of multiple _
        filename = filename.strip('_')  # remove trailing _
        filename = f"{filename}_{self.training_time}.{type}"
        filepath = os.path.join(path, filename)
        print(f"File to be saved: {filename}")
        return filepath

    @staticmethod
    def compute_accuracy(predictions: torch.Tensor,
                         targets: torch.Tensor) -> torch.Tensor:
        """Given a predicted and true labeling of an entire 3D image,
        returns the accuracy of the predicted labeling.

        Args:
            predictions: Output from model, shape: [B, L, H, W, D].
            targets: Ground truth, shape: [B, L, H, W, D].

        Returns:
            torch.Tensor: the accuracy of the prediction.
        """
        predictions_labeled = torch.argmax(predictions, dim=1)
        targets_labeled = torch.argmax(targets, dim=1)
        return torch.mean(predictions_labeled == targets_labeled, dtype=torch.float32)


trainer = Trainer()
trainer.train()
