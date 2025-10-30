import time

import numpy as np
import nibabel as nib
import torch
from monai.data import CacheDataset
from tqdm import tqdm
import os
from torch.utils.data import Dataset
import torchio as tio
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
        sample_dir_dict = [{'image': image_dir, 'label': label_dir} for image_dir, label_dir in self.sample_dirs]
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
        # Get the image and label
        image = nib.load(self.sample_dirs[idx][0]).get_fdata().astype(np.float32)
        label = nib.load(self.sample_dirs[idx][1]).get_fdata().astype(np.float32)

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


class ProMRIDataSetMonai(CacheDataset):
    """ Custom Dataset for the "Labelled weekly MR images of the male pelvis"

    Loads the MRI's from the .nii files into their image label pairs.

    Attributes:
        self.transformer: transformations to apply to each image
        self.image_paths: the 3d images of the MRI scans
        self.label_paths: the labels for the images
    """

    def __init__(self, sample_dirs, augment=False):
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
            num_workers=0,
            progress=True
        )

    def __getitem__(self, idx):
        data = super().__getitem__(idx)

        if self.augment:
            data = self.augmenter(data)


        return data["image"], data["label"]


class Transformer3D:
    def __init__(self):
        self.transforms = tio.Compose([
            tio.ZNormalization(),
            tio.Resize((128, 128, 64)),
            # decided by calculating the min, max value of each image, got the
            # average, min and max of all of them
            tio.Clamp(-0.75, 8.5),
            tio.RescaleIntensity(),
            # tio.Lambda(lambda x: x * (x > 0.01), types_to_apply=[tio.INTENSITY])
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
            tio.RandomAffine(
                scales=0,
                degrees=0.5),
            # tio.RandomElasticDeformation(max_displacement=07.5),
            tio.RandomGamma(
                log_gamma=(-0.3, 0.3)
            ),
            tio.RandomFlip(axes=1)
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


def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    """Code from [6]"""
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1

    return res

def one_hot_encode(labels: torch.Tensor) -> torch.Tensor:
    """Converts from [W, H, D] labels to [C, W, H, D] via one hot encoding, modified from [6].

    Args:
        labels: Initial labeling to one hot encode.

    Returns:
        torch.Tensor: one hot encoding of given labeling
    """
    print(f"onehot shape 1: {labels.shape}")
    labels = labels.squeeze()
    print(f"onehot shape 2: {labels.shape}")
    n_classes = torch.unique(labels)
    # one_hot = torch.nn.functional.one_hot(labels.long(), num_classes=n_classes)

    res = torch.zeros(labels.shape + (len(n_classes),), dtype=torch.float32)
    for c in n_classes:
        c = int(c)
        res[..., c:c+1][labels == c] = 1
    print(f"onehot shape 3: {res.shape}")
    res.permute(0, 4, 1, 2, 3)
    print(f"onehot shape 4: {res.shape}")
    return res