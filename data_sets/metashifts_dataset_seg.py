import os
from typing import Any, Callable, Optional, Tuple
from torchvision import datasets
from torchvision.datasets.folder import default_loader
import cv2
import numpy as np
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class MetaShiftSegDataset(datasets.ImageFolder):
    def __init__(
            self,
            data_path: str,
            image_sub_path: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            allow_empty: bool = False,
            mask_sub_path: Optional[str] = "train_masks_found",
    ):
        root = os.path.join(data_path, image_sub_path)
        super(MetaShiftSegDataset, self).__init__(root=root, transform=transform, target_transform=target_transform,
                                                  loader=loader, is_valid_file=is_valid_file, allow_empty=allow_empty)
        self.mask_sub_path = mask_sub_path
        self.mask_imgs = []
        for img in self.imgs:
            mask_im_path = img[0].replace(image_sub_path, mask_sub_path).replace('.jpg', '.png')
            self.mask_imgs.append(mask_im_path)
        self.num_classes = len(self.classes)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        mask_im = cv2.imread(self.mask_imgs[index], cv2.IMREAD_GRAYSCALE)
        mask_im = (mask_im > 0).astype('uint8')
        sample = self.loader(path)
        if self.transform is not None:
            transformed = self.transform(image=np.array(sample), mask=mask_im)
            im = transformed['image']
            mask_im = transformed['mask']
        return im, target, mask_im
