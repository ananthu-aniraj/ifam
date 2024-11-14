"""Class CUB200 from: https://github.com/zxhuang1698/interpretability-by-parts/"""
import os
import pandas as pd
import torch.utils.data
import torch
import torch.utils.data
from utils.data_utils.dataset_utils import load_json
from .fg_bird_dataset import FineGrainedBirdClassificationDataset


class CUBAttributesDataset(FineGrainedBirdClassificationDataset):
    """
    CUB200-2011 dataset with part attributes
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
        use_image_level_attributes, bool: Whether to use image level attributes or class level attributes.
    """

    def __init__(self, data_path, split=1, mode='train',
                 transform=None, image_sub_path="images", use_image_level_attributes=False,
                 load_non_part_attributes=False):
        super().__init__(data_path=data_path, split=split, mode=mode, transform=transform,
                         image_sub_path=image_sub_path)
        self.use_image_level_attributes = use_image_level_attributes
        self.load_non_part_attributes = load_non_part_attributes
        part_attribute_matcher = load_json(os.path.join(data_path, 'attributes', 'part_attribute_matcher.json'))
        # Sort the attributes by the part name alphabetically
        self.attributes_per_part = {part: part_attribute_matcher[part] for part in
                                    sorted(part_attribute_matcher.keys())}

        self.num_attributes_per_part_idx = [len(self.attributes_per_part[part_n]) for part_n in
                                            self.attributes_per_part.keys()]
        self.num_part_attributes = sum(self.num_attributes_per_part_idx)
        non_part_attributes = pd.read_csv(
            os.path.join(self.data_path, 'attributes', 'non_part_attributes.txt'), header=0)
        self.num_non_part_attributes = len(non_part_attributes.columns)
        self.total_num_attributes = self.num_part_attributes
        if self.load_non_part_attributes:
            self.total_num_attributes += self.num_non_part_attributes
        self.idx_to_part = {idx: part for idx, part in enumerate(self.attributes_per_part.keys())}

        if self.use_image_level_attributes:
            self._load_image_level_attributes()
        else:
            self._load_class_level_attributes()

    def _load_class_level_attributes(self):
        """
        Load class level attributes from the dataset
        """
        part_related_class_level_attributes = pd.read_csv(
            os.path.join(self.data_path, 'attributes', 'part_related_attributes.txt'), header=0)
        part_related_class_level_attributes = part_related_class_level_attributes.to_numpy()
        part_related_class_level_attributes = torch.from_numpy(part_related_class_level_attributes).float()

        non_part_related_class_level_attributes = pd.read_csv(
            os.path.join(self.data_path, 'attributes', 'non_part_attributes.txt'), header=0)
        non_part_related_class_level_attributes = non_part_related_class_level_attributes.to_numpy()
        non_part_related_attributes = torch.from_numpy(non_part_related_class_level_attributes).float()

        if self.load_non_part_attributes:
            self.visual_attributes = torch.hstack((part_related_class_level_attributes, non_part_related_attributes))
        else:
            self.visual_attributes = part_related_class_level_attributes

    def _load_image_level_attributes(self):
        """
        Load image level attributes from the dataset
        """
        image_level_part_only_attributes = pd.read_csv(
            os.path.join(self.data_path, 'attributes', 'image_level_part_only_attributes.csv'), header=0)
        image_level_part_only_attributes = image_level_part_only_attributes.loc[
            image_level_part_only_attributes['id'].isin(self.ids)]
        # Remove unnecessary columns
        image_level_part_only_attributes = image_level_part_only_attributes.drop(
            columns=['attribute_idx'])
        image_level_part_only_attributes = image_level_part_only_attributes[
            "is_present"].to_numpy().reshape(-1, self.num_part_attributes)
        image_level_part_only_attributes = torch.from_numpy(image_level_part_only_attributes).float()

        image_level_non_part_attributes = pd.read_csv(
            os.path.join(self.data_path, 'attributes', 'image_level_non_part_attributes.csv'), header=0)
        image_level_non_part_attributes = image_level_non_part_attributes.loc[
            image_level_non_part_attributes['id'].isin(self.ids)]
        # Remove unnecessary columns
        image_level_non_part_attributes = image_level_non_part_attributes.drop(
            columns=['attribute_idx'])
        image_level_non_part_attributes = image_level_non_part_attributes["is_present"].to_numpy().reshape(-1,
                                                                                                           self.num_non_part_attributes)
        image_level_non_part_attributes = torch.from_numpy(image_level_non_part_attributes).float()

        if self.load_non_part_attributes:
            self.visual_attributes = torch.hstack((image_level_part_only_attributes, image_level_non_part_attributes))
        else:
            self.visual_attributes = image_level_part_only_attributes

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_path, self.image_sub_path, self.names[idx])
        im = self.loader(image_path)
        label = self.labels[idx]
        if self.transform:
            im = self.transform(im)
        if self.use_image_level_attributes:
            visual_attributes = self.visual_attributes[idx]
        else:
            visual_attributes = self.visual_attributes[label]

        return im, visual_attributes, label
