import os
import numpy as np
import pandas as pd
import torch
from utils import pil_loader


class TurtleReID(torch.utils.data.Dataset):
    """
    A class for the Turtle ReID dataset.
    Variables
    data_path, str: Root directory of the dataset.
    mode, str: Current data split.
        "train": Training split
        "val": Validation split
        "test": Testing split
    transform, callable: A function/transform that takes in a PIL.Image and transforms it.
    image_sub_path, str: Path to the folder containing the images.
    split, str: Column name in metadata_base.csv to split the dataset.
    Available splits: 'split_closed', 'split_closed_random', 'split_open'
    """
    def __init__(self, data_path, split='train', transform=None, image_sub_path="images", split_type="split_closed"):
        self.data_path = data_path
        self.split = split
        self.transform = transform
        self.image_sub_path = image_sub_path
        self.loader = pil_loader
        self.split_type = split_type
        self.data = pd.read_csv(os.path.join(data_path, 'metadata_base.csv'))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.data['identity'].unique())}
        self.idx_to_class = {i: cls for i, cls in enumerate(self.data['identity'].unique())}
        self.class_names = self.data['identity'].unique()
        self.num_classes = len(self.data['identity'].unique())
        # Rename 'valid' to 'val'
        self.data = self.data.replace({'valid': 'val'})

        self.data = self.data.loc[self.data[split_type] == split]
        self.image_paths = self.data['file_name'].tolist()
        classes_present = self.data['identity'].tolist()

        self.labels = [self.class_to_idx[cls] for cls in classes_present]
        self.labels = np.array(self.labels)
        self.per_class_count = {self.class_to_idx[cls]: len(self.data[self.data['identity'] == cls]) for cls in self.class_names}
        self.cls_num_list = [self.per_class_count[idx] for idx in range(self.num_classes)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_path, self.image_sub_path, self.image_paths[idx])
        img = self.loader(image_path)
        label = self.labels[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label
