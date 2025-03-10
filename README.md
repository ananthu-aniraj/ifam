# Inherently Faithful Attention Maps for Vision Transformers

Implementation of the paper "Inherently Faithful Attention Maps for Vision Transformers" 

# Abstract
We introduce an attention-based method that uses learned binary attention masks to ensure that only attended image regions influence the prediction. Context can strongly affect object perception, sometimes leading to biased representations, particularly when objects appear in out-of-distribution backgrounds. At the same time, many image-level object-centric tasks require identifying relevant regions, often requiring context. To address this conundrum, we propose a two-stage framework: stage 1 processes the full image to discover object parts and identify task-relevant regions, while stage 2 leverages input attention masking to restrict its receptive field to these regions, enabling a focused analysis while filtering out potentially spurious information. Both stages are trained jointly, allowing stage 2 to refine stage 1. Extensive experiments across diverse benchmarks demonstrate that our approach significantly improves robustness against spurious correlations and out-of-distribution backgrounds.
# Setup
To install the required packages, run the following command:
```conda env create -f environment.yml```

Otherwise, you can also individually install the following packages:
1. [PyTorch](https://pytorch.org/get-started/locally/): Tested upto version 2.5.1.
2. [Colorcet](https://colorcet.holoviz.org/getting_started/index.html)
3. [Matplotlib](https://matplotlib.org/stable/users/installing.html)
4. [OpenCV](https://pypi.org/project/opencv-python-headless/)
5. [Pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html)
6. [Scikit-Image](https://scikit-image.org/docs/stable/install.html)
7. [Scikit-Learn](https://scikit-learn.org/stable/install.html) 
8. [TorchMetrics](https://torchmetrics.readthedocs.io/en/latest/pages/install.html)
9. [timm](https://pypi.org/project/timm/)
10. [wandb](https://pypi.org/project/wandb/): It is recommended to create an account and use it for tracking the experiments. Use the '--wandb' flag when running the training script to enable this feature.
11. [pycocotools](https://pypi.org/project/pycocotools/)
12. [pytopk](https://pypi.org/project/pytopk/)
13. [huggingface-hub](https://pypi.org/project/huggingface-hub/)
14. [pydicom](https://pydicom.github.io/pydicom/stable/tutorials/installation.html)
15. [albumentations](https://albumentations.ai/docs/getting_started/installation/)

# Datasets
### CUB
The dataset can be downloaded from [here](https://www.vision.caltech.edu/datasets/cub_200_2011/). 
The segmentation masks can be downloaded from [here](https://data.caltech.edu/records/w9d68-gec53) for the Foreground mIoU evaluation.

The folder structure should look like this:

```
CUB_200_2011
├── attributes
├── bounding_boxes.txt
├── classes.txt
├── images
├── image_class_labels.txt
├── images.txt
├── parts
├── segmentations
├── README
└── train_test_split.txt
```

### Waterbirds
The dataset can be downloaded from [here](https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz).
Extract it into the CUB_200_2011 folder to get the following structure:

```
CUB_200_2011
|── waterbird_complete95_forest2water2
├── attributes
├── bounding_boxes.txt
├── classes.txt
├── images
├── image_class_labels.txt
├── images.txt
├── parts
├── segmentations
├── README
└── train_test_split.txt
```


### Metashifts (Cat vs Dog)
Follow the instructions from this repository to download the Metashifts sub-set [here](https://github.com/Wuyxin/DISC?tab=readme-ov-file).

The folder structure should look like this:

```
MetaDatasetCatDog
|── test
|   |── cat
|   |   |── cat(shelf)
|   |── dog
|   |   |── dog(shelf)
|── train
|   |── cat
|   |   |── cat(bed)
|   |   |── cat(sofa)
|   |── dog
|   |   |── dog(bench)
|   |   |── dog(bike)
```

### SIIM-ACR Pneumothorax Segmentation
The dataset can be downloaded from [here](https://www.kaggle.com/datasets/jesperdramsch/siim-acr-pneumothorax-segmentation-data/data).
The pkl file containing chest tube presence annotations can be downloaded from [here](https://github.com/khaledsaab/spatial_specificity/blob/main/cxr_tube_dict.pkl).


After downloading the dataset, run the provided script to extract the images and masks.
``` python prepare_siim_acr.py --pkl_file_path <path to the pkl file> --root_dir <path where images are stored>```

The folder structure should look like this:

```
(root folder)
|── all_masks
|── full_test_set
|── train_set
|── robust_test_set
└── non_robust_test_set
```

### ImageNet-1k
The dataset can be downloaded from [here](https://image-net.org/download.php). Follow the instructions in this blog post to download the dataset: [link](https://medium.com/@billpsomas/download-and-prepare-imagenet-401bf10a681).
The folder structure should look like this:

```
(root folder)
|── train
|   |── n01440764
|   |   |── n01440764_10026.JPEG
|   |   |── n01440764_10027.JPEG
|   |── n01443537
|   |   |── n01443537_1000.JPEG
|   |   |── n01443537_1001.JPEG
|── val
|   |── n01440764
|   |   |── ILSVRC2012_val_00000293.JPEG
|   |   |── ILSVRC2012_val_00002138.JPEG
|   |── n01443537
|   |   |── ILSVRC2012_val_00000001.JPEG
|   |   |── ILSVRC2012_val_00000002.JPEG
```

### ImageNet-9
The dataset can be downloaded from [here](https://github.com/MadryLab/backgrounds_challenge/releases/download/data/backgrounds_challenge_data.tar.gz). The JSON file to map the class names to the ImageNet-1k classes can be downloaded from [here](https://github.com/MadryLab/backgrounds_challenge/blob/master/in_to_in9.json).
The data will be downloaded in the correct folder structure. No additional steps are required.


# Training
The details of running the training script can be found in the [training instructions](training_instructions.md) file.

# Evaluation
The details of running the evaluation metrics can be found in the [evaluation instructions](eval_instructions.md) file.


# Issues and Questions
Feel free to raise an issue if you face any problems with the code or have any questions about the paper.
