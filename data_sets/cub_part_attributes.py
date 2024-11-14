"""Class CUB200 from: https://github.com/zxhuang1698/interpretability-by-parts/"""
import os
import pandas as pd
import torch.utils.data
import numpy as np
import cv2
import torch
import torch.utils.data
from collections import defaultdict
from utils.data_utils.dataset_utils import pil_loader, load_json
from utils.misc_utils import file_line_count

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class CUBPartAttributesDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, split=1, mode='train', transform=None, image_sub_path="images",
                 use_image_level_attributes=False, load_non_part_attributes=False, relevant_part_name="head"):
        self.data_path = data_path
        self.mode = mode
        self.transform = transform
        self.image_sub_path = image_sub_path
        self.loader = pil_loader
        self.use_image_level_attributes = use_image_level_attributes
        self.relevant_part_name = relevant_part_name
        self.load_non_part_attributes = load_non_part_attributes
        train_test = pd.read_csv(os.path.join(data_path, 'train_test_split.txt'), sep='\s+',
                                 names=['id', 'train'])
        image_names = pd.read_csv(os.path.join(data_path, 'images.txt'), sep='\s+',
                                  names=['id', 'filename'])
        labels = pd.read_csv(os.path.join(data_path, 'image_class_labels.txt'), sep='\s+',
                             names=['id', 'label'])
        image_parts = pd.read_csv(os.path.join(data_path, 'parts', 'part_locs.txt'), sep='\s+',
                                  names=['id', 'part_id', 'x', 'y', 'visible'])
        bounding_boxes = pd.read_csv(os.path.join(data_path, 'bounding_boxes.txt'), sep='\s+',
                                     names=['id', 'x', 'y', 'width', 'height'])
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
        self.parts = image_parts
        self.num_classes = len(np.unique(self.labels))
        self.per_class_count = defaultdict(int)
        for label in self.labels:
            self.per_class_count[label] += 1
        self.cls_num_list = [self.per_class_count[idx] for idx in range(self.num_classes)]
        self.num_kps = file_line_count(os.path.join(data_path, 'parts', 'parts_merged_final.txt'))
        part_to_merged_parts = load_json(os.path.join(data_path, 'parts', 'parts_to_merged_parts.json'))
        # Convert the part index to the merged part index
        self.parts['part_id'] = self.parts['part_id'].apply(lambda x: part_to_merged_parts[str(x)])

        # Load the attributes
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

        self.use_albumentations = False

        # Find part idx for relevant part
        if self.relevant_part_name == "all":
            self.relevant_part_idx = 0
        elif self.relevant_part_name in self.idx_to_part.values():
            relevant_part_idx = [idx for idx, part in self.idx_to_part.items() if part == relevant_part_name][0]
            self.relevant_part_idx = relevant_part_idx + 1

            # Only store part locations for relevant part
            self.parts = self.parts.loc[self.parts['part_id'] == self.relevant_part_idx]
        else:
            raise ValueError(f"Part {self.relevant_part_name} not found in dataset")

        self.parts = self.parts.drop(columns=['part_id'])

        # Remove image ids that do not have the relevant part
        relevant_im_ids = self.parts[self.parts['visible'] == 1]['id'].values
        self.ids = np.array([im_id for im_id in self.ids if im_id in relevant_im_ids])
        relevant_im_names = dataset[dataset['id'].isin(relevant_im_ids)]['filename'].values
        self.names = np.array([im_name for im_name in self.names if im_name in relevant_im_names])

        relevant_im_labels = dataset[dataset['id'].isin(relevant_im_ids)]['label'].values
        labels_to_array = relevant_im_labels
        labels_to_index = {label: i for i, label in enumerate(np.unique(labels_to_array))}
        self.labels = np.array([labels_to_index[label] for label in labels_to_array])
        self.new_to_orig_label = {i: label for i, label in enumerate(np.unique(labels_to_array))}
        parts = self.parts[self.parts['id'].isin(relevant_im_ids)]
        self.parts = parts.to_numpy()
        self.bounding_boxes = bounding_boxes[bounding_boxes['id'].isin(relevant_im_ids)].drop(columns=['id']).to_numpy()
        self.category_id_to_name_bbox = {0: 'bird'}
        self.seg_masks = [name.replace('.jpg', '.png') for name in self.names]

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

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_path, self.image_sub_path, self.names[idx])
        im = self.loader(image_path)
        label = self.labels[idx]
        part_locs = self.parts[self.parts[:, 0] == self.ids[idx]][:, 1:]
        mask_im = cv2.imread(os.path.join(self.data_path, 'segmentations', self.seg_masks[idx]), cv2.IMREAD_GRAYSCALE)
        mask_im = (mask_im > 0).astype(np.uint8)
        # If there is more than one part location, randomly select one if visible
        if part_locs.shape[0] > 1:
            visible_part_locs = part_locs[part_locs[:, -1] == 1]
            part_locs = visible_part_locs[:, :-1]
        else:
            part_locs = part_locs[:, :-1].reshape(1, -1)
        bounding_boxes = self.bounding_boxes[idx].tolist()
        bounding_box_label = [0]  # All bounding boxes are labelled as bird
        if self.transform:
            transformed = self.transform(image=np.array(im), keypoints=part_locs, bboxes=[bounding_boxes], mask=mask_im,
                                         category_ids=bounding_box_label)
            im = transformed['image']
            part_locs = np.array(transformed['keypoints'])
            mask_im = transformed['mask'].unsqueeze(0)
            if part_locs.shape[0] > 1:
                part_locs = part_locs[np.random.choice(part_locs.shape[0], 1, replace=False)].reshape(-1)
            else:
                part_locs = part_locs.reshape(-1)
            part_locs = torch.from_numpy(part_locs).float()
            # Normalize the part locations to be between 0 and 1
            part_locs /= im.shape[1]
            # print(part_locs.shape)
        if self.use_image_level_attributes:
            visual_attributes = self.visual_attributes[idx]
        else:
            visual_attributes = self.visual_attributes[label]

        return im, visual_attributes, part_locs, mask_im, label
