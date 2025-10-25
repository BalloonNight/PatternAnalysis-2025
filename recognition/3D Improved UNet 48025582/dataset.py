import numpy as np
import nibabel as nib
import torch
from tqdm import tqdm
import os
from torch.utils.data import Dataset

# --------------------------- Define ProMRIDataSet --------------------------- #

class ProMRIDataSet(Dataset):
    """ Custom Dataset for the "Labelled weekly MR images of the male pelvis"

    Loads the MRI's from the .nii files into their image label pairs.

    Attributes:
        self.transformer: transformations to apply to each image
        self.image_paths: the 3d images of the MRI scans
        self.label_paths: the labels for the images
        self.patch_size: the size of the 3d patches to be returned
    """

    def __init__(self, image_dir, label_dir, transformer=None):
        self.transformer = transformer
        self.image_paths = sorted([os.path.join(image_dir, filename) for
                                   filename in os.listdir(image_dir)])
        self.label_paths = sorted([os.path.join(label_dir, filename) for
                                   filename in os.listdir(label_dir)])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the image and label
        image = nib.load(self.image_paths[idx]).get_fdata().astype(np.float32)
        label = nib.load(self.label_paths[idx]).get_fdata().astype(np.uint8)
        print(label.shape)

        # one hot encoding
        label = to_channels(label, np.uint8)
        label = np.moveaxis(label, 3, 0)

        # normalise
        image = (image - image.mean()) / image.std()

        # convert to tensor
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.uint8)).squeeze(0)
        print(label.shape)

        sample = {"image": image, "label": label}
        if self.transformer:
            self.transformer(sample)

        return sample


class Transformer3D:
    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # stuff like random flips

        sample['image'], sample['label'] = image, label

        return sample


# ------------------------ Appendix B Helper Functions ----------------------- #
# The following functions are all from the COMP3710_Report_v1.64_Final.pdf

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1

    return res

def load_data_3d(imageNames, normImage=False, categorical=False,
                 dtype=np.float32, getAffines=False, orient=False,
                 early_stop=False):
    """
    Load medical image data from names, cases list provided into a list for
    each. This function pre - allocates 5D arrays for conv3d to avoid excessive
     memory usage.

    Args:
        normImage: bool (normalise the image 0.0 -1.0).
        orient: Apply orientation and resample image? Good for images with large
            slice thickness or anisotropic resolution.
        dtype: Type of the data. If dtype = np.uint8, it is assumed that the
            data is labels.
        early_stop: Stop loading pre-maturely? Leaves arrays mostly empty, for
            quick loading and testing scripts.
    """
    affines = []

    #~ interp = 'continuous'
    interp = 'linear'
    if dtype == np . uint8 : # assume labels
        interp = 'nearest'

    # get fixed size
    num = len(imageNames)
    niftiImage = nib.load(imageNames[0])
    # if orient:
        # niftiImage = im.applyOrientation(niftiImage, interpolation=interp, scale=1)
        #~ testResultName = "oriented.nii.gz"
        #~ niftiImage.to_filename(testResultName)
    first_case = niftiImage.get_fdata(caching='unchanged')
    if len(first_case.shape) == 4:
        first_case = first_case[:,:,:,0] # sometimes extra dims, remove
    if categorical:
        first_case = to_channels(first_case, dtype = dtype)
        rows, cols, depth, channels = first_case.shape
        images = np.zeros((num ,rows ,cols ,depth ,channels ), dtype=dtype)
    else:
        rows, cols, depth = first_case . shape
        images = np.zeros((num, rows, cols, depth), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        # if orient:
            # niftiImage = im.applyOrientation(niftiImage, interpolation=interp, scale =1)
        inImage = niftiImage.get_fdata(caching='unchanged') # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 4:
            inImage = inImage[:, :, :, 0] # sometimes extra dims in HipMRI_study data
        inImage = inImage[:, :, :depth] # clip slices
        inImage = inImage.astype(dtype)
        if normImage:
            #~ inImage = inImage / np.linalg.norm(inImage)
            #~ inImage = 255. * inImage / inImage.max()
            inImage = (inImage - inImage.mean()) / inImage.std()
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            # ~ images [i ,: ,: ,: ,:] = inImage
            images[i, :inImage.shape[0], :inImage.shape[1], :inImage.shape[2], :inImage.shape[3]] = inImage # with pad
        else:
            # ~ images [i ,: ,: ,:] = inImage
            images [i, :inImage . shape [0] ,: inImage . shape [1] ,: inImage . shape [2]] = inImage # with pad

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images