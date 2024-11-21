import torch
import os
from data_sets import FineGrainedBirdClassificationDataset, CXRDataset, WaterBirdsDataset
from torchvision import datasets


def get_dataset(args, train_transforms, test_transforms):
    if args.dataset == 'cub':
        dataset_train = FineGrainedBirdClassificationDataset(args.data_path, split=args.train_split, mode='train',
                                                             transform=train_transforms,
                                                             image_sub_path=args.image_sub_path_train)
        dataset_test = FineGrainedBirdClassificationDataset(args.data_path, mode=args.eval_mode,
                                                            transform=test_transforms,
                                                            image_sub_path=args.image_sub_path_test)
        num_cls = dataset_train.num_classes

    elif args.dataset == 'waterbirds':
        dataset_train = WaterBirdsDataset(args.data_path, mode='train',
                                          transform=train_transforms, image_sub_path=args.image_sub_path_train)
        dataset_test = WaterBirdsDataset(args.data_path, mode=args.eval_mode,
                                         transform=test_transforms, image_sub_path=args.image_sub_path_test)
        num_cls = dataset_train.num_classes

    elif args.dataset == 'meta_shift':
        dataset_train = datasets.ImageFolder(
            os.path.join(args.data_path, args.image_sub_path_train),
            train_transforms)
        dataset_test = datasets.ImageFolder(os.path.join(args.data_path, args.image_sub_path_test), test_transforms)
        num_cls = len(dataset_test.classes)
        dataset_test.num_classes = num_cls
        dataset_train.num_classes = num_cls
    elif args.dataset == 'siim_acr':
        dataset_train = CXRDataset(args.data_path, image_sub_path=args.image_sub_path_train,
                                   mask_sub_path=args.mask_sub_path, transform=train_transforms)
        dataset_test = CXRDataset(args.data_path, image_sub_path=args.image_sub_path_test,
                                  mask_sub_path=args.mask_sub_path, transform=test_transforms)
        num_cls = dataset_train.num_classes
    else:
        raise ValueError('Dataset not supported.')
    return dataset_train, dataset_test, num_cls
