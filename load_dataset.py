import torch
import os
from data_sets import FineGrainedBirdClassificationDataset, CXRDataset, WaterBirdsDataset, TurtleReID, ImageNetV2Folder, \
    ImageNetSubSetEval
from torchvision import datasets


def load_train_test_datasets(args, train_transforms, test_transforms):
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
    elif args.dataset == 'turtle_reid':
        dataset_train = TurtleReID(args.data_path, split='train', transform=train_transforms,
                                   image_sub_path=args.image_sub_path_train, split_type=args.split_type)
        dataset_test = TurtleReID(args.data_path, split=args.eval_mode, transform=test_transforms,
                                  image_sub_path=args.image_sub_path_test, split_type=args.split_type)
        num_cls = dataset_train.num_classes
    elif args.dataset == 'imagenet_a' or args.dataset == 'imagenet_r':
        dataset_train = ImageNetSubSetEval(
            os.path.join(args.data_path, args.image_sub_path_train),
            train_transforms, variant=args.dataset)
        dataset_test = ImageNetSubSetEval(os.path.join(args.data_path, args.image_sub_path_test), test_transforms,
                                          variant=args.dataset)
        num_cls = 1000
    elif args.dataset == 'imagenet':
        dataset_train = datasets.ImageFolder(root=os.path.join(args.data_path, args.image_sub_path_train),
                                             transform=train_transforms)
        dataset_test = datasets.ImageFolder(root=os.path.join(args.data_path, args.image_sub_path_test),
                                            transform=test_transforms)
        num_cls = len(dataset_test.classes)
        dataset_test.num_classes = num_cls
        dataset_train.num_classes = num_cls
    elif args.dataset == 'imagenet_v2':
        dataset_train = ImageNetV2Folder(os.path.join(args.data_path, args.image_sub_path_train),
                                         transform=train_transforms)
        dataset_test = ImageNetV2Folder(os.path.join(args.data_path, args.image_sub_path_test),
                                        transform=test_transforms)
        num_cls = len(dataset_test.classes)
        dataset_test.num_classes = num_cls
        dataset_train.num_classes = num_cls
    else:
        raise ValueError('Dataset not supported.')
    return dataset_train, dataset_test, num_cls


def load_single_split(args, image_transforms, split='test', image_sub_path='images'):
    if args.dataset == 'cub':
        dataset_test = FineGrainedBirdClassificationDataset(args.data_path, split=split,
                                                            transform=image_transforms,
                                                            image_sub_path=image_sub_path)
        num_cls = dataset_test.num_classes

    elif args.dataset == 'waterbirds':
        dataset_test = WaterBirdsDataset(args.data_path, split=split,
                                         transform=image_transforms, image_sub_path=image_sub_path)
        num_cls = dataset_test.num_classes

    elif args.dataset == 'meta_shift':
        dataset_test = datasets.ImageFolder(os.path.join(args.data_path, image_sub_path), image_transforms)
        num_cls = len(dataset_test.classes)
        dataset_test.num_classes = num_cls
    elif args.dataset == 'siim_acr':
        dataset_test = CXRDataset(args.data_path, image_sub_path=image_sub_path,
                                  mask_sub_path=args.mask_sub_path, transform=image_transforms)
        num_cls = dataset_test.num_classes
    elif args.dataset == 'turtle_reid':
        dataset_test = TurtleReID(args.data_path, split=split, transform=image_transforms,
                                  image_sub_path=image_sub_path, split_type=args.split_type)
        num_cls = dataset_test.num_classes
    elif args.dataset == 'imagenet_a' or args.dataset == 'imagenet_r':
        dataset_test = ImageNetSubSetEval(os.path.join(args.data_path, image_sub_path), image_transforms,
                                          variant=args.dataset)
        num_cls = 1000
    elif args.dataset == 'imagenet':
        dataset_test = datasets.ImageFolder(root=os.path.join(args.data_path, image_sub_path),
                                            transform=image_transforms)
        num_cls = len(dataset_test.classes)
        dataset_test.num_classes = num_cls
    elif args.dataset == 'imagenet_v2':
        dataset_test = ImageNetV2Folder(os.path.join(args.data_path, image_sub_path),
                                        transform=image_transforms)
        num_cls = len(dataset_test.classes)
        dataset_test.num_classes = num_cls
    else:
        raise ValueError('Dataset not supported.')
    return dataset_test, num_cls
