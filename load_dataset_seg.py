from data_sets import CUBDatasetSeg, WaterBirdDatasetSeg, MetaShiftSegDataset


def get_dataset_seg(args, train_transforms, test_transforms):
    if args.dataset == 'cub':
        dataset_train = CUBDatasetSeg(args.data_path, split=args.train_split, mode='train',
                                      transform=train_transforms,
                                      image_sub_path=args.image_sub_path_train,
                                      mask_sub_path=args.mask_sub_path_train)
        dataset_test = CUBDatasetSeg(args.data_path, mode=args.eval_mode,
                                     transform=test_transforms,
                                     image_sub_path=args.image_sub_path_test,
                                     mask_sub_path=args.mask_sub_path_test)
        num_cls = dataset_train.num_classes
    elif args.dataset == 'waterbirds':
        dataset_train = WaterBirdDatasetSeg(args.data_path, mode='train',
                                            transform=train_transforms, image_sub_path=args.image_sub_path_train,
                                            mask_sub_path=args.mask_sub_path_train)
        dataset_test = WaterBirdDatasetSeg(args.data_path, mode=args.eval_mode,
                                           transform=test_transforms, image_sub_path=args.image_sub_path_test,
                                           mask_sub_path=args.mask_sub_path_test)
        num_cls = dataset_train.num_classes
    elif args.dataset == 'meta_shift':
        dataset_train = MetaShiftSegDataset(data_path=args.data_path, image_sub_path=args.image_sub_path_train,
                                            mask_sub_path=args.mask_sub_path_train, transform=train_transforms)
        dataset_test = MetaShiftSegDataset(data_path=args.data_path, image_sub_path=args.image_sub_path_test,
                                           mask_sub_path=args.mask_sub_path_test, transform=test_transforms)
        num_cls = dataset_train.num_classes
    else:
        raise ValueError('Dataset not supported.')
    return dataset_train, dataset_test, num_cls
