#!/usr/bin/env python3
"""train.py
An implementation of training the 3D Improved UNet3D module.

Reference for [4] and [5] can be found in README.md
"""
import os
import re
import time
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
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
        self.batch_size = 4
        self.num_epochs = 300
        self.on_cluster = False
        self.learning_rate = 5e-4
        self.lr_schedule = 0.985
        self.weight_decay = 1e-5
        self.scale_data_size = 1
        self.validate_split = 0.1
        self.test_split = 0.1
        self.smooth = 1e-6
        self.patience = 10

        # Device Config
        self.device = torch.device('cuda' if torch.cuda.is_available() else
                                   'cpu')
        print(f'Using device: {self.device}')

        # Components
        self.train_loader = None
        self.test_loader = None
        self.validate_loader = None
        self.init_data_loaders()
        self.model = md.ImprovedUNet3D(1, self.n_labels, base_channels=8)

    def init_data_loaders(self):
        """Initialize the data loaders"""
        print("defining data loaders...")

        # get image paths
        if self.on_cluster:
            image_paths, label_paths = ds.get_data_path(self.rangpur_scan_path, self.rangpur_label_path)
        else:
            image_paths, label_paths = ds.get_data_path(self.pc_scan_path, self.pc_label_path)

        # zip together teh lists
        samples = list(zip(image_paths, label_paths))

        # determine dataset sizes
        dataset_length = int(len(samples) * self.scale_data_size)
        validate_size = int(self.validate_split * dataset_length)
        test_size = int(self.test_split * dataset_length)
        train_size = dataset_length - validate_size - test_size
        unused_size = len(samples) - train_size - validate_size - test_size
        data_lengths = [train_size, validate_size, test_size, unused_size]

        # split up samples
        train_dirs, validate_dirs, test_dirs, unused_dirs = ds.random_split(samples, data_lengths)

        print(
            f"data loaders being created (using "
            f"{sum(data_lengths[:2])}/{len(samples)}, "
            f"{(sum(data_lengths[:2]) / len(samples)) * 100:.4f}%):\n"
            f"    data langths: {data_lengths}\n"
            f"    train loader: size={len(train_dirs)}\n"
            f"    validate loader: size={len(validate_dirs)}\n"
            f"    test loader: size={len(test_dirs)}\n"
            f"    unsued loader: size={len(unused_dirs)}"
        )

        train_set = ds.ProMRIDataSetMonai(train_dirs, augment=True)
        validate_set = ds.ProMRIDataSetMonai(validate_dirs)
        test_set = ds.ProMRIDataSetMonai(test_dirs)

        # DataLoaders
        self.train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True)
        self.validate_loader = DataLoader(
            validate_set,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True)
        self.test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True)
        total = len(self.train_loader) + len(self.validate_loader) + len(self.test_loader)
        print(
            f"data loaders created (using "
            f"{total}/{len(samples)}, "
            f"{(total / len(samples)) * 100:.4f}%):\n"
            f"    train loader: size={len(self.train_loader)}\n"
            f"    validate loader: size={len(self.validate_loader)}\n"
            f"    test loader: size={len(self.test_loader)}\n"
            f"    unsued loader: size={len(unused_dirs)}"
        )

    def train(self):
        """Train the model, code modified from [4]"""
        print("Training 3D Improved UNet3D...")
        self.model.to(self.device)
        torch.backends.cudnn.benchmark = True
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.learning_rate,
                                     weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_schedule)

        # Tracking times
        start_time = time.time()
        next_vis = 1
        checkin_interval = 60
        checkin_time = time.time() + checkin_interval
        epoch_times = []

        # Early stop
        best_loss = 1
        counter = 0

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
            print(f'Epoch [{epoch + 1}/{self.num_epochs}], Patience [{counter}/{self.patience}]:')
            epoch_start_time = time.time()
            # Training loop with progress
            self.model.train()
            multi_dice_loss = 0
            dice_coefficients = [0, 0, 0, 0, 0, 0, 0]
            accuracy = 0
            for batch_idx, (images, true_labels) in enumerate(
                    self.train_loader):
                images = images.to(self.device)
                true_labels = true_labels.to(self.device)

                # Forwards pass
                optimizer.zero_grad()

                # Get the prediction and its dice loss
                # with torch.amp.autocast('cuda'):
                predict_labels = self.model(images)
                cur_loss, cur_coefficients = self.multiclass_dice_loss(predict_labels, true_labels)

                """# Reshape from [B, C, H, W, D] to [H, W, D]
                image = images[0, 0, :, :, :].cpu().numpy()
                true_label = torch.argmax(true_labels, dim=1)[0, :, :, :
                ].cpu().numpy()
                predict_label = torch.argmax(predict_labels, dim=1)[0, :, :, :
                ].cpu().numpy()
                title = f"traning Batch testing {batch_idx}"
                self.plot_3d_image(image, true_label, predict_label, title, cur_coefficients)"""

                # Backward pass
                cur_loss.backward()
                optimizer.step()

                # update current epoches loss's
                multi_dice_loss += cur_loss.item()
                dice_coefficients = [x + y for x, y in zip(dice_coefficients,
                                                           cur_coefficients)]
                accuracy += self.compute_accuracy(predict_labels, true_labels)

                # Check in
                if time.time() >= checkin_time:
                    print(f"    Checkin:\n"
                          f"        time passed: {time.time() - start_time:.4f}\n"
                          f"        batch_inx: {batch_idx}\n"
                          f"        curr_loss: {multi_dice_loss / (batch_idx + 1):.4f}\n"
                          f"        curr_coefficient: {[f'{c / (batch_idx + 1):.4f}' for c in dice_coefficients]}\n"
                          f"        accuracy: {accuracy / (batch_idx + 1):.4f}")
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
                  f"        Coefficients: {[f'{c:.4f}' for c in avg_coefficients]}\n"
                  f"        Accuracy: {avg_accuracy:.4f}")

            # Validation Loop
            self.model.eval()
            multi_dice_loss = 0
            dice_coefficients = [0, 0, 0, 0, 0, 0, 0]
            accuracy = 0
            worst_coefficient = 1
            with torch.no_grad():
                for batch_idx, (images, true_labels) in enumerate(
                        self.validate_loader):
                    # load data
                    images = images.to(self.device)
                    true_labels = true_labels.to(self.device)

                    # predict
                    predict_labels = self.model(images)

                    # get losses
                    cur_loss, cur_coefficients = self.multiclass_dice_loss(predict_labels, true_labels)
                    multi_dice_loss += cur_loss.item()
                    dice_coefficients = [x + y for x, y in zip(
                        dice_coefficients, cur_coefficients)]
                    accuracy += self.compute_accuracy(predict_labels,
                                                      true_labels)

                    # track worst coefficient
                    if worst_coefficient > min(cur_coefficients[:-1]):
                        worst_coefficient = min(cur_coefficients[:-1])

                    # Check in
                    if time.time() >= checkin_time:
                        print(f"    Checkin:\n"
                              f"        time passed: {time.time() - start_time:.4f}\n"
                              f"        batch_inx: {batch_idx}\n"
                              f"        curr_loss: {multi_dice_loss / (batch_idx + 1):.4f}\n"
                              f"        curr_coefficient: {[f'{c / (batch_idx + 1):.4f}' for c in dice_coefficients]}\n"
                              f"        accuracy: {accuracy / (batch_idx + 1):.4f}")
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
                  f"        Coefficients: {[f'{c:.4f}' for c in avg_coefficients]}\n"
                  f"        Accuracy: {avg_accuracy:.4f}")

            # step the learning rate scheduling
            scheduler.step()

            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            print(f"    Epoch Time: {epoch_time:.4f}, Average Epoch Time: {sum(epoch_times) / len(epoch_times)}")

            # Early stop
            if best_loss > avg_loss:
                counter = 0
                best_loss = avg_loss
            else:
                counter += 1

            if counter > self.patience:
                break

            # Visualize
            if (epoch + 1) >= next_vis or (epoch + 1) == 0 or (epoch + 1) == self.num_epochs:
                vis_start_time = time.time()
                self.show_predictions((epoch + 1))
                next_vis *= 2
                print(
                    f"    Visualization Time: "
                    f"{time.time() - vis_start_time:.4f} seconds")

        end = time.time()
        elapsed = end - start_time
        print(f"Training time: {elapsed/60:.4f} minutes")

        # plot training data
        self.show_graphs(train_losses, train_coefficients, train_accuracy, validate_losses, validate_coefficients, validate_accuracy)

        # test training data
        worst_coefficient = self.test()

        title = f"Final Model {worst_coefficient}"
        self.save_model(title)

    def test(self, visualise: bool = False) -> float:
        """Tests the current loaded model.

        Args:
            visualise: if the visuals should also be printed.

        Returns:
            float: The worst coefficient.
        """
        self.model.eval()

        start_time = time.time()
        checkin_interval = 60
        checkin_time = time.time() + checkin_interval

        multi_dice_loss = 0
        dice_coefficients = [0, 0, 0, 0, 0, 0, 0]
        accuracy = 0
        worst_coefficient = 1
        with (torch.no_grad()):
            for batch_idx, (images, true_labels) in enumerate(
                    self.test_loader):
                # load data
                images = images.to(self.device)
                true_labels = true_labels.to(self.device)

                # predict
                predict_labels = self.model(images)

                # get losses
                cur_loss, cur_coefficients = self.multiclass_dice_loss(
                    predict_labels, true_labels)
                multi_dice_loss += cur_loss.item()
                dice_coefficients = [x + y for x, y in zip(
                    dice_coefficients, cur_coefficients)]
                accuracy += self.compute_accuracy(predict_labels,
                                                  true_labels)

                for coefficient in dice_coefficients:
                    if worst_coefficient > coefficient:
                        worst_coefficient = coefficient

                # Reshape from [B, C, H, W, D] to [H, W, D]
                image = images[0, 0, :, :, :].cpu().numpy()
                true_label = torch.argmax(true_labels, dim=1)[0, :, :, :
                ].cpu().numpy()
                predict_label = torch.argmax(predict_labels, dim=1)[0, :, :, :
                ].cpu().numpy()
                title = f"Testing Batch {batch_idx}"
                self.plot_3d_image(image, true_label, predict_label, title, cur_coefficients, visualise=visualise)

                # Check in
                if time.time() >= checkin_time:
                    print(f"    Checkin:\n"
                          f"        time passed: {time.time() - start_time:.4f}\n"
                          f"        batch_inx: {batch_idx}\n"
                          f"        curr_loss: {multi_dice_loss / (batch_idx + 1):.4f}\n"
                          f"        curr_coefficient: {[f'{c / (batch_idx + 1):.4f}' for c in dice_coefficients]}\n"
                          f"        accuracy: {accuracy / (batch_idx + 1):.4f}")
                    checkin_time = time.time() + checkin_interval

        # update training losses
        avg_loss = multi_dice_loss / len(self.test_loader)
        avg_coefficients = [x / len(self.test_loader) for x in
                            dice_coefficients]
        avg_accuracy = accuracy / len(self.test_loader)
        print(f"    Testing:\n"
              f"        Loss: {avg_loss:.4f}\n"
              f"        Coefficients: {[f'{c:.4f}' for c in avg_coefficients]}\n"
              f"        Accuracy: {avg_accuracy:.4f}")

        return worst_coefficient

    def show_graphs(self, train_losses, train_coefficients, train_accuracy, validate_losses, validate_coefficients, validate_accuracy):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        title = f"Graphed Training Data"
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # dice loss
        axes[0, 0].plot(train_losses, label="Train Losses")
        axes[0, 0].plot(validate_losses, label="Validate Losses")
        axes[0, 0].set_title("Dice loss")
        axes[0, 0].set_xlabel("EPOCH")
        axes[0, 0].set_ylabel("Dice Loss")
        axes[0, 0].legend()

        # accuracy
        axes[1, 0].plot(train_accuracy, label="Train Accuracy")
        axes[1, 0].plot(validate_accuracy, label="Validate Accuracy")
        axes[1, 0].set_title("Accuracy")
        axes[1, 0].set_xlabel("EPOCH")
        axes[1, 0].set_ylabel("Accuracy")
        axes[1, 0].legend()

        # train coefficients
        background, body, bones, bladder, rectum, prostate, multiclass = zip(*train_coefficients)
        axes[0, 1].plot(background, label="Background")
        axes[0, 1].plot(body, label="Body")
        axes[0, 1].plot(bones, label="Bones")
        axes[0, 1].plot(bladder, label="Bladder")
        axes[0, 1].plot(rectum, label="Rectum")
        axes[0, 1].plot(prostate, label="Prostate")
        axes[0, 1].plot(multiclass, label="Multiclass")
        axes[0, 1].set_title("Training Dice Similarity Coefficient")
        axes[0, 1].set_xlabel("EPOCH")
        axes[0, 1].set_ylabel("Dice Similarity Coefficient")
        axes[0, 1].legend(fontsize="small")

        # validation coefficients
        background, body, bones, bladder, rectum, prostate, multiclass = zip(*validate_coefficients)
        axes[1, 1].plot(background, label="Background")
        axes[1, 1].plot(body, label="Body")
        axes[1, 1].plot(bones, label="Bones")
        axes[1, 1].plot(bladder, label="Bladder")
        axes[1, 1].plot(rectum, label="Rectum")
        axes[1, 1].plot(prostate, label="Prostate")
        axes[1, 1].plot(multiclass, label="Multiclass")
        axes[1, 1].set_title("Validate Dice Similarity Coefficient")
        axes[1, 1].set_xlabel("EPOCH")
        axes[1, 1].set_ylabel("Dice Similarity Coefficient")
        axes[1, 1].legend(fontsize="small")

        # save
        plt.tight_layout()
        save_path = self.get_formatted_filepath(self.image_save_path, title, "png")
        plt.savefig(save_path)
        plt.close()

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

            _, coefficients = self.multiclass_dice_loss(predict_label, true_label.to(self.device))
            # Reshape from [B, C, H, W, D] to [H, W, D]
            image = image[0, 0, :, :, :].cpu().numpy()
            true_label = torch.argmax(true_label, dim=1)[0, :, :, :
            ].cpu().numpy()
            predict_label = torch.argmax(predict_label, dim=1)[0, :, :, :
            ].cpu().numpy()

            # Plotting
            title = f"Epoch_{epoch}_Prediction"
            self.plot_3d_image(image, true_label, predict_label, title, coefficients)

    def multiclass_dice_loss(self, predictions: torch.Tensor,
                             targets: torch.Tensor) -> tuple[float, list[float]]:
        """Calculates the Dice Coefficient of each label in the prediction, tensors must
        be in the shape [B, L, H, W, D]. Code modified from [4].

        Dice Loss = 1 - Dice Coefficient.
        Dice Coefficient = (2/|K|) * sum((sum(u, v) / (sum(u) + sum(v))).

        Args:
            predictions: Output from model, shape: [B, L, H, W, D]
            targets: Ground truth, shape: [B, L, H, W, D]

        Returns:
            tuple[float, list[float]]: List of each individual labels and the multiclass Dice coefficient.
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
            dice_coefficients.append(dice_coefficient.item())

        avg_dice_coefficient = total_dice_coefficient / self.n_labels
        dice_coefficients.append(avg_dice_coefficient.item())

        # Return Dice Loss (1 - Dice Coefficient)
        return (1 - avg_dice_coefficient), dice_coefficients

    def plot_3d_image(self, image: np.ndarray, true_label: np.ndarray,
                      predict_label: np.ndarray, title: str, dcs_values: list[float], visualise: bool = False):
        """Plots a 3d image by scrolling through one of its axis. modified from
        [5]. The images must be of shape [H, W, D].

        Args:
            image: The 3D image to be plotted.
            true_label: The true labeling for the MRI scan.
            predict_label: The predicted labeling for the MRI scan.
            title: The title to be put into the graph and file save name.
            dcs_values: List of all the dice coefficients for each label.
             in order [Background, Body, Bone, Bladder, Rectum, Prostate, Multiclass].
            visualise: True if the plot should be printed to the screen.
        """
        # flip to wanted orientation
        image = np.flip(np.transpose(image, axes=[0, 2, 1]), axis=1)
        true_label = np.flip(np.transpose(true_label, axes=[0, 2, 1]), axis=1)
        predict_label = np.flip(np.transpose(predict_label, axes=[0, 2, 1]),
                                axis=1)

        # make and label the subplots
        fig, axes = plt.subplots(4, figsize=(6, 12))
        axes[0].set_title(f"Image")
        axes[1].set_title(f"Real Label")
        axes[2].set_title(f"Predict Label")

        # starting point for each graph
        imshow1 = axes[0].imshow(image[0], cmap='gray', vmin=np.min(image),
                                 vmax=np.max(image), animated=True)
        imshow2 = axes[1].imshow(true_label[0], cmap='tab10', vmin=0, vmax=5,
                                 animated=True)
        imshow3 = axes[2].imshow(predict_label[0], cmap='tab10', vmin=0, vmax=5,
                                 animated=True)

        # update function
        def update(frame: int) -> list:
            """Given the frame index, returns the frames do display in each subplot.

            Args:
                frame: the frame index

            Returns:
                list: list of the frames for each subplot.
            """
            imshow1.set_array(image[frame])
            imshow2.set_array(true_label[frame])
            imshow3.set_array(predict_label[frame])
            return [imshow1, imshow2, imshow3]

        # make the animator and title
        ani = animation.FuncAnimation(fig, update, frames=image.shape[0],
                                      interval=1, blit=True, repeat=True)


        # add dcs values
        axes[3].axis("off")
        dcs_labels = ["Background", "Body", "Bone", "Bladder", "Rectum", "Prostate", "Multiclass"]
        dcs_text = "\n".join([f"{label}: {value:.2f}" for label, value in zip(dcs_labels, dcs_values)])
        axes[3].text(0, 0.5, f"DCS Scores:\n{dcs_text}", fontsize=18, va='center', ha='left')

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()

        # save the gif
        writer = animation.PillowWriter(fps=120, metadata=dict(artist='Me'),
                                        bitrate=1800)
        save_path = self.get_formatted_filepath(self.image_save_path, title, "gif")
        ani.save(save_path, writer=writer, dpi=80)
        if visualise:
            plt.show()
        plt.close()

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

    def save_model(self, model_name):
        """Save the current model.

        Args:
            model_name: The name to save the current model as.
        """
        filename = self.get_formatted_filepath(self.model_save_path, model_name, "pt")
        torch.save(self.model, filename)

    def load_model(self, model_path):
        """Given the name of a model, loads it.

        Args:
            model_path: The name of the model to load into the class.
        """
        self.model = torch.load(model_path, map_location=self.device).to(self.device)

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
        return torch.mean((predictions_labeled == targets_labeled).float()).item()

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
