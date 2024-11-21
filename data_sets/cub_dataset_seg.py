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


class CUBDatasetSeg(torch.utils.data.Dataset):
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

    def __init__(self, data_path, split=1, mode='train', transform=None, image_sub_path="images"):
        self.data_path = data_path
        self.mode = mode
        self.transform = transform
        self.image_sub_path = image_sub_path
        self.loader = pil_loader
        train_test = pd.read_csv(os.path.join(data_path, 'train_test_split.txt'), sep='\s+',
                                 names=['id', 'train'])
        image_names = pd.read_csv(os.path.join(data_path, 'images.txt'), sep='\s+',
                                  names=['id', 'filename'])
        labels = pd.read_csv(os.path.join(data_path, 'image_class_labels.txt'), sep='\s+',
                             names=['id', 'label'])
        image_parts = pd.read_csv(os.path.join(data_path, 'parts', 'part_locs.txt'), sep='\s+',
                                  names=['id', 'part_id', 'x', 'y', 'visible'])
        dataset = train_test.merge(image_names, on='id')
        dataset = dataset.merge(labels, on='id')

        if mode == 'train':
            dataset = dataset.loc[dataset['train'] == 1]
            samples_train = np.arange(len(dataset))
            self.train_samples = samples_train[:int(len(samples_train) * split)]
            dataset = dataset.iloc[self.train_samples]
        elif mode == 'test':
            dataset = dataset.loc[dataset['train'] == 0]
        elif mode == 'val':
            dataset = dataset.loc[dataset['train'] == 1]
            samples_val = np.arange(len(dataset))
            self.val_samples = samples_val[int(len(samples_val) * split):]
            dataset = dataset.iloc[self.val_samples]

        # training images are labelled 1, test images labelled 0. Add these
        # images to the list of image IDs
        self.ids = dataset['id'].to_numpy()
        self.names = dataset['filename'].to_numpy()
        # Handle the case where the labels are not 0-indexed and there are gaps
        labels_to_array = dataset['label'].to_numpy()
        labels_to_index = {label: i for i, label in enumerate(np.unique(labels_to_array))}
        self.labels = np.array([labels_to_index[label] for label in labels_to_array])
        self.new_to_orig_label = {i: label for i, label in enumerate(np.unique(labels_to_array))}
        image_parts = image_parts.loc[image_parts['id'].isin(self.ids)]
        self.parts = image_parts[image_parts['visible'] == 1]
        self.num_classes = len(np.unique(self.labels))
        self.per_class_count = defaultdict(int)
        for label in self.labels:
            self.per_class_count[label] += 1
        self.cls_num_list = [self.per_class_count[idx] for idx in range(self.num_classes)]
        self.seg_masks = [name.replace('.jpg', '.png') for name in self.names]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_path, self.image_sub_path, self.names[idx])
        mask_im = cv2.imread(os.path.join(self.data_path, 'segmentations', self.seg_masks[idx]), cv2.IMREAD_GRAYSCALE)
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
