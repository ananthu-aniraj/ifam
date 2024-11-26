import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from data_sets import FineGrainedBirdClassificationDataset, WaterBirdsDataset, CXRDataset
from load_model import load_model_2_stage
import argparse
from tqdm import tqdm
import copy
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.training_utils.engine_utils import load_state_dict_snapshot
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# fix all the randomness for reproducibility
torch.backends.cudnn.enabled = True
torch.manual_seed(0)
torch.cuda.manual_seed(0)

torch.set_float32_matmul_precision('high')
torch._dynamo.config.capture_scalar_outputs = True


def parse_args():
    parser = argparse.ArgumentParser(description='Inference benchmark models')
    parser.add_argument('--model_arch', default='resnet50', type=str,
                        help='pick model architecture')
    parser.add_argument('--use_torchvision_resnet_model', default=False, action='store_true')
    parser.add_argument('--use_hf_transformers', default=False, action='store_true')
    # Data
    parser.add_argument('--data_path',
                        help='directory that contains cub files, must'
                             'contain folder "./images"', required=True)
    parser.add_argument('--image_sub_path', default='images', type=str, required=False)
    parser.add_argument('--dataset', default='cub', type=str)
    parser.add_argument('--anno_path_test', default='', type=str, required=False)

    parser.add_argument('--mask_sub_path', default='all_masks', type=str, required=False)
    # Model params
    parser.add_argument('--num_parts', help='number of parts to predict',
                        default=8, type=int)
    parser.add_argument('--image_size', default=448, type=int)
    parser.add_argument('--output_stride', default=32, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    # Part Dropout
    parser.add_argument('--part_dropout', default=0.0, type=float)
    parser.add_argument('--part_dropout_stage_2', default=0.0, type=float)
    # Drop path
    parser.add_argument('--drop_path_stage_1', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--drop_path_stage_2', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    # Gumbel Softmax
    parser.add_argument('--gumbel_softmax', default=False, action='store_true')
    parser.add_argument('--softmax_temperature', default=1.0, type=float)

    # Model path
    parser.add_argument('--model_path', default=None, type=str)
    # Use soft masks for second stage
    parser.add_argument('--use_soft_masks', default=False, action='store_true')
    # Use part logit thresholds (only for evaluation)
    parser.add_argument('--part_logits_threshold_path', default="", type=str)

    # Torch compile mode
    parser.add_argument('--torch_compile_mode', default='max-autotune-no-cudagraphs', type=str)

    args = parser.parse_args()
    return args


def benchmark(args):
    args.eval_only = True
    args.pretrained_start_weights = True
    height = args.image_size
    test_transforms = transforms.Compose([
        transforms.Resize(size=height),
        transforms.CenterCrop(size=height),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])
    # define dataset path
    if args.dataset == 'cub':
        cub_path = args.data_path
        # define dataset and loader
        eval_data = FineGrainedBirdClassificationDataset(cub_path, split=1, transform=test_transforms, mode='test')
        num_cls = eval_data.num_classes
    elif args.dataset == 'waterbirds':
        eval_data = WaterBirdsDataset(args.data_path, mode='test',
                                      transform=test_transforms, image_sub_path=args.image_sub_path)
        num_cls = eval_data.num_classes
    elif args.dataset == 'meta_shift':
        eval_data = datasets.ImageFolder(os.path.join(args.data_path, args.image_sub_path), test_transforms)
        num_cls = len(eval_data.classes)
        eval_data.num_classes = num_cls
    elif args.dataset == 'siim_acr':
        cxr_mean = (0.48865, 0.48865, 0.48865)
        cxr_std = (0.24621, 0.24621, 0.24621)
        test_transforms: A.Compose = A.Compose([
            A.ToRGB(always_apply=True),
            A.Resize(args.image_size, args.image_size),
            A.Normalize(mean=cxr_mean, std=cxr_std),
            ToTensorV2()
        ])
        eval_data = CXRDataset(args.data_path, image_sub_path=args.image_sub_path_test,
                               mask_sub_path=args.mask_sub_path, transform=test_transforms)
        num_cls = eval_data.num_classes
    else:
        raise ValueError('Dataset not supported.')
    # Load the model
    model = load_model_2_stage(args, eval_data, num_cls)
    snapshot_data = torch.load(args.model_path, map_location=torch.device('cpu'), weights_only=True)
    if 'model_state' in snapshot_data:
        _, state_dict = load_state_dict_snapshot(snapshot_data)
    else:
        state_dict = copy.deepcopy(snapshot_data)
    model.load_state_dict(state_dict, strict=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.eval().to(device)
    model = torch.compile(model, mode=args.torch_compile_mode, dynamic=True)
    test_loader = torch.utils.data.DataLoader(
        eval_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    images = None
    # Warmup
    for i, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc='Warmup'):
        images = batch[0].to(device)
        with torch.inference_mode():
            output = model(images)
        if i == 300:
            break

    # Benchmark
    for idx in tqdm(range(300), desc="Inference benchmark"):
        with torch.inference_mode():
            output = model(images)

    print("Inference benchmark done!")

    torch._dynamo.reset()


if __name__ == '__main__':
    args = parse_args()
    benchmark(args)
