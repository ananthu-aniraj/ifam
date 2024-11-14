import os
from torchvision import datasets
import pandas as pd
from collections import defaultdict
from utils.data_utils.dataset_utils import pil_loader


def load_class_mapping(class_list_file):
    with open(class_list_file) as f:
        class_index_to_class_name = {i: line.strip() for i, line in enumerate(f)}
    return class_index_to_class_name


def load_species_mapping(species_map_file):
    df = pd.read_csv(species_map_file, sep=';', quoting=1, dtype={'species_id': str})
    df = df.set_index('species_id')
    return df['species'].to_dict()


class PlantCLEF2024(datasets.ImageFolder):
    """
    Class to train models on PlantCLEF2024
    Variables
        base_folder, str: Root directory of the dataset.
        image_sub_path, str: Path to the folder containing the images.
        class_mapping, str: Path to the file containing the class mapping.
        species_mapping, str: Path to the file containing the species mapping.
        transform, callable: A function/transform that takes in a PIL.Image and transforms it.
    """
    def __init__(self, base_folder, image_sub_path, transform=None, target_transform=None, loader=pil_loader):
        root = os.path.join(base_folder, image_sub_path)
        super().__init__(root, transform=transform, target_transform=target_transform, loader=loader)
        class_mapping_path = os.path.join(base_folder, 'class_mapping.txt')
        species_mapping_path = os.path.join(base_folder, 'species_id_to_name.txt')
        self.species_mapping = load_species_mapping(species_mapping_path)
        self.idx_to_class = load_class_mapping(class_mapping_path)
        new_class_to_idx = {class_name: idx for idx, class_name in self.idx_to_class.items()}
        original_class_to_idx = self.class_to_idx
        original_idx_to_class = {idx: class_name for class_name, idx in original_class_to_idx.items()}
        self.class_to_idx = new_class_to_idx
        self.classes = list(self.idx_to_class.values())
        self.targets = [self.class_to_idx[original_idx_to_class[target]] for target in self.targets]
        self.samples = [(path, self.class_to_idx[original_idx_to_class[target]]) for path, target in self.samples]
        self.imgs = self.samples
        self.num_classes = len(self.classes)
        self.per_class_count = {class_name: 0 for class_name in self.classes}
        for target in self.targets:
            self.per_class_count[self.idx_to_class[target]] += 1
        self.cls_num_list = [self.per_class_count[class_name] for class_name in self.classes]


