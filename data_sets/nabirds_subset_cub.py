import os
import pandas as pd
import numpy as np
import torch
import glob
from collections import defaultdict
from utils.data_utils.dataset_utils import pil_loader


class NABirdsCUBSubset(torch.utils.data.Dataset):
    """
    A class for evaluating CUB models on the NABirds dataset.
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

    def __init__(self, data_path, transform=None, image_sub_path="nabirds_test_set"):
        self.data_path = data_path
        self.transform = transform
        self.image_sub_path = image_sub_path
        self.loader = pil_loader
        classes = pd.read_csv(os.path.join(data_path, 'classes.txt'), sep='\s+',
                              names=['label', 'class_name'])
        # Make labels 0-indexed
        classes['label'] = classes['label'] - 1
        classes_dict = {row['class_name']: row['label'] for _, row in classes.iterrows()}
        img_files = glob.glob(os.path.join(data_path, image_sub_path, '**', '*.jpg'), recursive=True)
        img_label_names = [os.path.basename(os.path.dirname(img_file)) for img_file in img_files]
        labels = [classes_dict[label_name] for label_name in img_label_names]
        self.labels = np.array(labels)
        self.names = img_files
        self.num_classes = 200
        self.per_class_count = defaultdict(int)
        for label in self.labels:
            self.per_class_count[label] += 1
        self.cls_num_list = [self.per_class_count[idx] for idx in range(self.num_classes)]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.names[idx]
        im = self.loader(image_path)
        label = self.labels[idx]

        if self.transform:
            im = self.transform(im)

        return im, label
