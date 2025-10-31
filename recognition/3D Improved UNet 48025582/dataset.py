#!/usr/bin/env python3
"""dataset.py
An implementation for the loading and handling the dataset.
"""
from monai.data import CacheDataset
import os
import monai.transforms as mt
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
        index += length

    return sub_data


class ProMRIDataSetMonai(CacheDataset):
    """ Custom Dataset for the "Labelled weekly MR images of the male pelvis"

    Loads the MRI's from the .nii files into their image label pairs.

    Attributes:
        self.augment: Stores if augments will randomly occur
        self.augmenter: Stores the random augments
    """

    def __init__(self, sample_dirs, augment=False):
        """Initializes the dataset.

        Args:
            sample_dirs: The directories of all the samples.
            augment: If this class should randomly augment its data.
        """
        # restructure dirs into dict
        sample_dirs = [{'image': image_dir, 'label': label_dir} for image_dir, label_dir in sample_dirs]
        self.augment = augment

        # define transformations
        transformer = mt.Compose([
            mt.LoadImaged(keys=["image", "label"],
                          image_only=True),
            mt.EnsureChannelFirstd(keys=["image", "label"]),
            mt.Resized(keys=["image", "label"],
                       spatial_size=(128, 128, 64),
                       mode=["trilinear", "nearest"]),
            mt.NormalizeIntensityd(keys=["image"],
                                   nonzero=True),
            mt.ScaleIntensityd(keys=["image"]),
            mt.AsDiscreted(keys=["label"], to_onehot=6),
            mt.ToTensorD(keys=["image", "label"])
        ])
        self.augmenter = mt.Compose([
            mt.RandAffined(keys=["image", "label"],
                           prob=0.5,
                           rotate_range=(0.1, 0.1, 0.1),
                           mode=["bilinear", "nearest"],
                           padding_mode="zeros"),
            mt.RandGaussianNoised(keys=["image"], std=0.01)
        ])

        # cache
        super().__init__(
            data=sample_dirs,
            transform=transformer,
            cache_rate=1.0,
            num_workers=4,
            progress=True
        )

    def __getitem__(self, idx: int) -> tuple:
        """Get sample data at a particular index.

        Args:
            idx: The Index of the sample data to get.

        Returns:
            tuple: image data and label data
        """
        data = super().__getitem__(idx)

        if self.augment:
            data = self.augmenter(data)


        return data["image"], data["label"]