import os
import pandas as pd
import torch
from collections import defaultdict

from utils.data_utils.dataset_utils import pil_loader


class WaterBirdsDataset(torch.utils.data.Dataset):
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

    def __init__(self, data_path, mode='train', transform=None,
                 image_sub_path="waterbird_complete95_forest2water2"):
        self.data_path = data_path
        self.mode = mode
        self.transform = transform
        self.image_sub_path = image_sub_path
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

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_path, self.image_sub_path, self.names[idx])
        im = self.loader(image_path)
        label = self.labels[idx]
        place_label = self.place_labels[idx]
        if self.transform:
            im = self.transform(im)

        return im, label, place_label


if __name__ == '__main__':
    pass
