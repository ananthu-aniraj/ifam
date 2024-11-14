# Ref: https://github.com/khaledsaab/spatial_specificity/blob/main/src/data/cxr.py
import os
import glob

import numpy as np
import torch
import pydicom
import cv2


class CXRDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, image_sub_path, mask_sub_path, transform=None):
        """

        Args:
            data_path: Path to the dataset
            image_sub_path: Sub path to the images
            mask_sub_path: Sub path to the masks
            transform: Transform to be applied on a sample
        """
        self.data_path = data_path
        self.image_sub_path = image_sub_path
        self.transform = transform
        self.mask_sub_path = mask_sub_path

        image_paths = glob.glob(os.path.join(self.data_path, self.image_sub_path, "**/*.dcm"), recursive=True)
        self.image_paths = sorted(image_paths)
        labels = [os.path.basename(os.path.dirname(image_path)) for image_path in self.image_paths]
        self.labels = [int(label) for label in labels]
        self.labels = np.array(self.labels)
        self.num_classes = 2

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def load_image(image_path):
        img = pydicom.dcmread(image_path).pixel_array
        # Convert to 8-bit
        img = img.astype(np.uint8)
        return img

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_name = image_path.split('/')[-1].split('.dcm')[0]
        img = self.load_image(image_path)
        label = self.labels[idx]
        mask = cv2.imread(os.path.join(self.data_path, self.mask_sub_path, image_name + '.png'),
                          cv2.IMREAD_GRAYSCALE)
        mask[mask > 0] = 1
        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

        return img, label, mask
