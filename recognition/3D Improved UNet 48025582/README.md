# Segmentation of the Prostate 3D Dataset using 3D Improved UNet3D
**Name:** Bailey Renshaw\
**Student Number:** 48025582\
**Task:** 7 [Hard Difficulty - 3D Improved UNet]

## Description
This project implements the 3D Improved UNet3D model from [2] to segment the 
"MR images of the male pelvis" from [1]. Particularly, it is capable of 
successfully segmenting the MRI's to label their background, body, bones, 
bladder, rectum and prostate.

This type of segmentation can be used for assisting with:
 - Helping know and plant treatments around organ boundaries.
 - Identifying abnormal organ boundaries.
 - Monitoring changes in organ boundaries over time.
 - More consistent labeling compared to several humans.
 - Can be trained to notice the smallest details that might be missed by a 
 - human.

Roughly, the 3D Improved UNet3D model improves upon the UNet model by 
implementing residual paths, context modules, localization modules and 
segmentation layers. This all assists in its ability to better understand the 
context of a large 3D image.

## How 3D Improved UNet3D Works
![](./assets/Improved_UNet3D_Structure.png)
*Figure 1: 3D Improved UNet3D Model Structure Diagram from [1].*

Here you can se the internal structure for the model as descibed by [1], my 
implementation is slightly different as I used half as many output filters 
which gave a much better performance without sacrificing on much learning.

### Context Pathway
 - *Context Modules*: Pre-activated residual blocks with dropout.
 - *Convolutions*: General convolutions where all but the first double the 
number of output filters for the next layer.

This downsampling leads to the capturing of high level context.

### Localization Pathway
 - *Localization Modules*: These are similar to the context modules but 
designed to localize small structures
 - *Upsampling Module*: This halves the feature map.

This upsampling leads to the capturing of low level structures.

### Segmentation Layers
The segmentation layers are designed to implement deep supervision into the 
localisation pathway. each localisation layer turns the filters at its layer 
into a simple 6 filter feature map and upscales the lower feature maps and 
combines them with the higher ones.

## Prostate 3D Dataset
This project uses the *"Labelled weekly MR images of the male pelvis"* dataset 
from [1] to train it's model.

The dataset specifications:
 - Week0 to Week7 of MRI scans on people undergoing prostate cancer radiation 
therapy, where therapy starts Week1.
 - The labels [Background, Body, Bones, Bladder, Rectum, Prostate].
 - Collected between 2014-01-01 and 2014-12-31
 - 211 separate MRI scan and label pairs
 - Image dimensions are 256x256x128 Voxels
   - For this model this was halved down to 128x128x64 voxels for computational speed

And for training this data was split up into 80% training, 10% validation, and 
10% testing. to increase the theoretical size of the training set several 
augmentations took place, those being random rotation and random noise.

## Dependencies
Python 3.11.3
matplotlib==3.10.7
monai==1.5.1
nibabel==5.3.2
numpy==2.3.4
torch==2.5.1+cu121
torchio==0.20.23

## Usage
### Project structre
first off this following project structure needs to be followed, particularly 
the files for containing the data.
```
3D Improved UNet 48025582
> modules.py                                # Components of the model.
> dataset.py                                # Dataloader and pre-processing.
> train.py                                  # Training, validation, testing, saving, loading and plotting of the model.
> predict.py                                # Run an already created model.
> README.md                                 # Description of the project.
> semantic_MRs_anon/                        # This can technicly be changed in train.py.
----> Case_NNN_WeekK_LFOV.nii.gz            # The internal strucutre of the data must be the same.
> semantic_labels_anon/                     # This can technicly be changed in train.py.
----> Case_NNN_WeekK_SEMANTIC_LFOV.nii.gz   # The internal strucutre of the data must be the same.
> output/                                   # Outputs from running the model.
----> model/                                # Location of finished models.
----> results/                              # Location of plots and visualizations.
```
### Training the Model
train a model:
```
python train.py
```
this will simply train the model how i exactly did, however there are many 
hyperparameters inside to edit if you wish.

test a model:
```
python predict.py
```
this will run the test dataset on one of the finished models, to change which 
one change the file name inside the code.

### Training output
## How 3D Improved UNet3D Works
Here is the Dice Loss, Accuracy and Dice Similarity Coefficients of the 
training and validation data over time after running 133 Epochs:
![](./assets/Graphed_Training_Data_1761868372.png)

Below here is 3 random animated test set samples:

1 | 2 | 3
--- | --- | ---
![](./assets/Testing_Batch_1_1761868372.gif) | ![](./assets/Testing_Batch_3_1761868372.gif) | ![](./assets/Testing_Batch_5_1761868372.gif)

The Dice Similarity Coefficients of each label for each sample are is above 
0.8, thus completing the requirement of at least 0.7 DSC for each label of the 
task successfully.

## References
[1] J. Dowling, P. Greer, *"Labelled weekly MR images of the male pelvis"*, 2021, DOI: https://doi.org/10.25919/45t8-p065

[2] F. Isensee, P Kickingereder, W. Wick, M. Bendszus, K. H. Maier-Hein, *"Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge"*, 2018, DOI: https://doi.org/10.48550/arXiv.1802.10508

[3] P. K. Kao, *"Modified-3D-UNet-Pytorch"*, 2018, GitHub repository, [Online]. Available: https://github.com/pykao/Modified-3D-UNet-Pytorch

[4] W. Dai, *COMP3710, "UNet_segmentation_code_demo.ipynb"*, course materials, The University of Queensland, 2025, [Access limited to enrolled students].

[5] J. D. Hunter, *matplotlib.org, "Animated scatter saved as GIF"*, 2025, [Online]. Available: https://matplotlib.org/stable/gallery/animation/simple_scatter.html

[6] S. Chandra, *COMP3710, "Report Pattern Recognition"*, course materials, The University of Queensland, 2025, [Access limited to enrolled students].
