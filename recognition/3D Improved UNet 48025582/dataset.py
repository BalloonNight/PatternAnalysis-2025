import time

import numpy as np
import nibabel as nib
import torch
from tqdm import tqdm
import os
from torch.utils.data import Dataset
import torchio as tio
import random

def get_data_path(image_dir: str, label_dir: str) -> tuple[list[str], list[str]]:
    """Given the directories of the images and labels, gets the list of names for all of them.

    Args:
        image_dir: The directory of the images.
        label_dir: The directory of the labels.

    Returns:
        tuple[list[str], list[str]]: Returns the list of image paths and list of label paths.
    """
    image_paths = sorted([os.path.join(image_dir, filename) for
                               filename in os.listdir(image_dir)])
    label_paths = sorted([os.path.join(label_dir, filename) for
                               filename in os.listdir(label_dir)])
    return image_paths, label_paths

def random_split(data: list, lengths: list[int]) -> list[list]:
    """Given a list of data and a list of lengths, will randomly split the data up amongst those lengths.

    Args:
        data: Data to be split up.
        lengths: Lengths for the data to be split up into.

    Returns:
        list[list]: A list of sublists with random elements from data.
    """
    assert len(data) == sum(lengths)

    # randomise list
    shuffled = data.copy()
    random.shuffle(shuffled)

    # distribute list among sub lists
    sub_data = []
    index = 0
    for length in lengths:
        sub_data.append(shuffled[index:index + length])
        index += 1

    return sub_data


class ProMRIDataSet(Dataset):
    """ Custom Dataset for the "Labelled weekly MR images of the male pelvis"

    Loads the MRI's from the .nii files into their image label pairs.

    Attributes:
        self.transformer: transformations to apply to each image
        self.image_paths: the 3d images of the MRI scans
        self.label_paths: the labels for the images
    """

    def __init__(self, sample_dirs, device='cpu', augment=False, pre_load=True):
        self.transformer = Transformer3D()
        self.augmenter = Augmenter3D()
        self.augment = augment
        self.pre_load = pre_load
        self.device = device
        # Get a sorted list of all the files in the given directories
        self.sample_dirs = sample_dirs
        # Pre loading
        self.samples = []
        if pre_load:
            self.preload_data()

    def set_augment(self, augment: bool):
        self.augment = augment

    def preload_data(self):
        for idx in range(len(self.sample_dirs)):
            self.samples.append(self.load_sample(idx))
            if idx % 10 == 0:
                print(f"loaded: {idx}/{len(self.sample_dirs)}")

    def load_sample(self, idx):
        start_time = time.time()
        # Get the image and label
        image = nib.load(self.sample_dirs[idx][0]).get_fdata().astype(np.float32)
        label = nib.load(self.sample_dirs[idx][1]).get_fdata().astype(np.float32)

        start_time = time.time()
        # one hot encoding
        label = to_channels(label, np.float32)
        label = np.moveaxis(label, 3, 0)

        # convert to tensor
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        image = image.unsqueeze(0)

        # Perform given transformation
        image, label = self.transformer(image, label)

        return image, label

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        # Load the data (depending on if its been pre_loaded or not)
        if self.pre_load:
            image, label = self.samples[idx]
        else:
            image, label = self.load_sample(idx)

        # Augment if that's wanted
        if self.augment:
            image, label = self.augmenter(image, label)

        return image, label


class Transformer3D:
    def __init__(self):
        self.transforms = tio.Compose([
            tio.ZNormalization(),
            tio.Resize((128, 128, 64)),
            # decided by calculating the min, max value of each image, got the
            # average, min and max of all of them
            tio.Clamp(-0.75, 8.5),
            tio.RescaleIntensity(),
            tio.Lambda(lambda x: x * (x > 0.01), types_to_apply=[tio.INTENSITY])
        ])

    def __call__(self, image, label):
        # format for tio
        sample = tio.Subject(
            image = tio.ScalarImage(tensor=image),
            label = tio.LabelMap(tensor=label)
        )
        # apply generic transforms
        sample = self.transforms(sample)
        # apply augments if requested

        return sample['image'].data, sample['label'].data

class Augmenter3D:
    def __init__(self):
        self.augments = tio.Compose([
            tio.RandomAffine(),
            tio.RandomElasticDeformation(),
            tio.RandomGamma(),
            tio.RandomFlip(axes=(0, 1, 2))
        ])

    def __call__(self, image, label):
        # format for tio
        sample = tio.Subject(
            image = tio.ScalarImage(tensor=image),
            label = tio.LabelMap(tensor=label)
        )
        # apply augments
        sample = self.augments(sample)
        return sample['image'].data, sample['label'].data


# ------------------------ Appendix B Helper Functions ----------------------- #
# The following functions are all from the COMP3710_Report_v1.64_Final.pdf

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1

    return res