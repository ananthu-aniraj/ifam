import torch
from torchvision import transforms as transforms
from torchvision.transforms import Compose
import albumentations as A
from albumentations.pytorch import ToTensorV2
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform
from .planckian_color_jitter import PlanckianJitter


def make_train_transforms(args):
    train_transforms: Compose = transforms.Compose([
        transforms.Resize(size=args.image_size, antialias=True),
        transforms.RandomHorizontalFlip(p=args.hflip),
        transforms.RandomVerticalFlip(p=args.vflip),
        transforms.ColorJitter(),
        transforms.RandomApply([PlanckianJitter(mode="blackbody")], p=0.8),
        transforms.RandomAffine(degrees=90, translate=(0.2, 0.2), scale=(0.8, 1.2)),
        transforms.RandomCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

    ])
    return train_transforms


def make_test_transforms(args):
    test_transforms: Compose = transforms.Compose([
        transforms.Resize(size=args.image_size, antialias=True),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

    ])
    return test_transforms


def build_transform_timm(args, is_train=True):
    resize_im = args.image_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.image_size,
            is_training=True,
            color_jitter=args.color_jitter,
            hflip=args.hflip,
            vflip=args.vflip,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.image_size, padding=4)
        return transform

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if args.image_size >= 384:
            t.append(
                transforms.Resize((args.image_size, args.image_size),
                                  interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            )
            print(f"Warping {args.image_size} size input images...")
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.image_size / args.crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            )
            t.append(transforms.CenterCrop(args.image_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def inverse_normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    mean = torch.as_tensor(mean)
    std = torch.as_tensor(std)
    un_normalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    return un_normalize


def normalize_only(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    normalize = transforms.Normalize(mean=mean, std=std)
    return normalize


def inverse_normalize_w_resize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
                               resize_resolution=(256, 256)):
    mean = torch.as_tensor(mean)
    std = torch.as_tensor(std)
    resize_unnorm = transforms.Compose([
        transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist()),
        transforms.Resize(size=resize_resolution, antialias=True)])
    return resize_unnorm


def load_train_transforms_siim_acr(args):
    cxr_mean = (0.48865, 0.48865, 0.48865)
    cxr_std = (0.24621, 0.24621, 0.24621)
    train_transforms: A.Compose = A.Compose([
        A.ToRGB(),
        A.Resize(args.image_size, args.image_size),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=cxr_mean, std=cxr_std),
        ToTensorV2()
    ])
    test_transforms: A.Compose = A.Compose([
        A.ToRGB(),
        A.Resize(args.image_size, args.image_size),
        A.Normalize(mean=cxr_mean, std=cxr_std),
        ToTensorV2()
    ])
    return train_transforms, test_transforms


def load_transforms(args):
    # Get the transforms and load the dataset
    if args.dataset.lower().startswith('imagenet'):
        if args.augmentations_to_use == 'timm':
            train_transforms = build_transform_timm(args, is_train=True)
        else:
            # Default ImageNet transforms from torchvision
            train_transforms = transforms.Compose([
                transforms.RandomResizedCrop(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
            ])
        test_transforms = build_transform_timm(args, is_train=False)
        return train_transforms, test_transforms
    if args.dataset == 'siim_acr':
        train_transforms, test_transforms = load_train_transforms_siim_acr(args)
        return train_transforms, test_transforms
    if args.augmentations_to_use == 'timm':
        train_transforms = build_transform_timm(args, is_train=True)
    elif args.augmentations_to_use == 'cub_original':
        train_transforms = make_train_transforms(args)
    else:
        raise ValueError('Augmentations not supported.')
    test_transforms = make_test_transforms(args)
    return train_transforms, test_transforms


def load_transforms_seg(args):
    train_transforms = A.Compose([
        A.SmallestMaxSize(max_size=args.image_size),
        A.CropNonEmptyMaskIfExists(p=1, height=args.image_size, width=args.image_size),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.5),
        A.PlanckianJitter(p=0.5, mode="blackbody"),
        A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ToTensorV2()
    ])
    test_transforms = A.Compose([
        A.SmallestMaxSize(max_size=args.image_size),
        A.CropNonEmptyMaskIfExists(p=1, height=args.image_size, width=args.image_size),
        A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ToTensorV2()
    ])
    return train_transforms, test_transforms
