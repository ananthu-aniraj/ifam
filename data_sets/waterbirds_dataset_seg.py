"""Class CUB200 from: https://github.com/zxhuang1698/interpretability-by-parts/"""
import os
import pandas as pd
import numpy as np
import cv2
import torch
from collections import defaultdict
from utils.data_utils.dataset_utils import pil_loader

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class WaterBirdDatasetSeg(torch.utils.data.Dataset):
    """
    A general class for fine-grained bird classification datasets. Tested for CUB200-2011 and NABirds.
    Variables
    ----------
        data_path, str: Root directory of the dataset.
        split, int: Percentage of training samples to use for training.
        mode, str: Current data split.
            "train": Training split
            "val": Validation split
            "test": Testing split
        transform, callable: A function/transform that takes in a PIL.Image and transforms it.
        image_sub_path, str: Path to the folder containing the images.
    """

    def __init__(self, data_path, mode='train', transform=None, image_sub_path="waterbird_complete95_forest2water2", mask_sub_path="segmentations"):
        self.data_path = data_path
        self.mode = mode
        self.transform = transform
        self.image_sub_path = image_sub_path
        self.mask_sub_path = mask_sub_path
        self.loader = pil_loader
        dataset = pd.read_csv(os.path.join(data_path, image_sub_path, 'metadata.csv'))

        if mode == 'train':
            dataset = dataset.loc[dataset['split'] == 0]
        elif mode == 'val':
            dataset = dataset.loc[dataset['split'] == 1]
        else:
            dataset = dataset.loc[dataset['split'] == 2]
            if mode == 'worst_case':
                dataset_waterbird = dataset.loc[(dataset['y'] == 1) & (dataset['place'] == 0)]
                dataset_landbird = dataset.loc[(dataset['y'] == 0) & (dataset['place'] == 1)]
                dataset = pd.concat([dataset_waterbird, dataset_landbird])
            elif mode == 'best_case':
                dataset_waterbird = dataset.loc[(dataset['y'] == 1) & (dataset['place'] == 1)]
                dataset_landbird = dataset.loc[(dataset['y'] == 0) & (dataset['place'] == 0)]
                dataset = pd.concat([dataset_waterbird, dataset_landbird])
        # training images are labelled 0, test images labelled 2. Add these
        # images to the list of image IDs
        self.ids = dataset['img_id'].to_numpy()
        self.names = dataset['img_filename'].to_numpy()
        labels_to_array = dataset['y'].to_numpy()
        self.labels = labels_to_array
        self.place_labels = dataset['place'].to_numpy()
        self.num_classes = 2
        self.per_class_count = defaultdict(int)
        for label in self.labels:
            self.per_class_count[label] += 1
        self.cls_num_list = [self.per_class_count[idx] for idx in range(self.num_classes)]
        self.seg_masks = [name.replace('.jpg', '.png') for name in self.names]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_path, self.image_sub_path, self.names[idx])
        mask_im = cv2.imread(os.path.join(self.data_path, self.mask_sub_path, self.seg_masks[idx]), cv2.IMREAD_GRAYSCALE)
        mask_im = (mask_im > 0).astype(np.uint8)
        im = self.loader(image_path)
        label = self.labels[idx]

        if self.transform:
            transformed = self.transform(image=np.array(im), mask=mask_im)
            im = transformed['image']
            mask_im = transformed['mask']
        return im, label, mask_im


if __name__ == '__main__':
    pass
