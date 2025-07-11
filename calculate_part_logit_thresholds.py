import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from data_sets import FineGrainedBirdClassificationDataset, WaterBirdsDataset, CXRDataset
from load_model import load_model_2_stage
import argparse
from tqdm import tqdm
import copy
import os
from pathlib import Path
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.training_utils.engine_utils import load_state_dict_snapshot
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from utils import factors, save_json
from load_dataset import load_single_split

# fix all the randomness for reproducibility
torch.backends.cudnn.enabled = True
torch.manual_seed(0)
torch.cuda.manual_seed(0)


def parse_args():
    parser = argparse.ArgumentParser(description='Calculate part logit thresholds')
    parser.add_argument('--model_arch', default='resnet50', type=str,
                        help='pick model architecture')
    parser.add_argument('--use_torchvision_resnet_model', default=False, action='store_true')
    parser.add_argument('--use_hf_transformers', default=False, action='store_true')
    # Data
    parser.add_argument('--data_path',
                        help='directory that contains cub files, must'
                             'contain folder "./images"', required=True)
    parser.add_argument('--image_sub_path', default='images', type=str)
    parser.add_argument('--dataset', default='cub', type=str)
    parser.add_argument('--dataset_split', default='train', type=str)
    parser.add_argument('--mask_sub_path', default='all_masks', type=str, required=False)
    parser.add_argument('--split_type', default="split_closed", type=str, help='split of the dataset to use')

    # Model params
    parser.add_argument('--num_parts', help='number of parts to predict',
                        default=8, type=int)
    parser.add_argument('--image_size', default=518, type=int)
    parser.add_argument('--output_stride', default=32, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
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
    # Path to save the results
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--save_histograms', default=False, action='store_true')
    return parser.parse_args()


def calc_part_logits(args):
    args.eval_only = True
    args.pretrained_start_weights = True
    height = args.image_size
    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    if args.dataset == 'siim_acr':
        cxr_mean = (0.48865, 0.48865, 0.48865)
        cxr_std = (0.24621, 0.24621, 0.24621)
        image_transforms: A.Compose = A.Compose([
            A.ToRGB(),
            A.Resize(args.image_size, args.image_size),
            A.Normalize(mean=cxr_mean, std=cxr_std),
            ToTensorV2()
        ])
    else:
        image_transforms = transforms.Compose([
            transforms.Resize(size=height),
            transforms.CenterCrop(size=height),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ])
    # Load the dataset
    eval_data, num_cls = load_single_split(args, image_transforms, split=args.dataset_split, image_sub_path=args.image_sub_path)
    # Load the model
    model = load_model_2_stage(args, num_cls)
    snapshot_data = torch.load(args.model_path, map_location=torch.device('cpu'), weights_only=True)
    if 'model_state' in snapshot_data:
        _, state_dict = load_state_dict_snapshot(snapshot_data)
    else:
        state_dict = copy.deepcopy(snapshot_data)
    model.load_state_dict(state_dict, strict=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.eval().to(device)
    dataloader = torch.utils.data.DataLoader(
        eval_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # Calculate the part logits
    part_logits_train = []
    attn_maps_combined_train = []
    for data in tqdm(dataloader, desc="Processing"):
        images = data[0]
        images = images.to(device)

        with torch.no_grad():
            pred_stage_2, attn_maps_combined, attn_maps_fg_bg_combined, all_features_mod, pred_stage_1, part_logits = model(
                images)
            part_logits_train.append(part_logits.cpu())
            attn_maps_combined_train.append(attn_maps_combined.cpu())

    part_logits_train = torch.cat(part_logits_train, dim=0)
    attn_maps_combined_train = torch.cat(attn_maps_combined_train, dim=0)
    part_assignments = torch.argmax(attn_maps_combined_train, dim=1)  # get the part assignments
    part_logits_full_matched = {}
    for idx, (logits, assignment) in enumerate(zip(part_logits_train, part_assignments)):
        for part_idx in range(attn_maps_combined_train.shape[1]):
            if part_idx not in part_logits_full_matched:
                part_logits_full_matched[part_idx] = []
            part_logits_full_matched[part_idx].append(logits[part_idx][assignment == part_idx])

    for key in part_logits_full_matched.keys():
        part_logits_full_matched[key] = torch.cat(part_logits_full_matched[key], dim=0)

    part_logits_full_unmatched = {}
    for idx, (logits, assignment) in enumerate(zip(part_logits_train, part_assignments)):
        for part_idx in range(attn_maps_combined_train.shape[1]):
            if part_idx not in part_logits_full_unmatched:
                part_logits_full_unmatched[part_idx] = []
            part_logits_full_unmatched[part_idx].append(logits[part_idx][assignment != part_idx])

    for key in part_logits_full_unmatched.keys():
        part_logits_full_unmatched[key] = torch.cat(part_logits_full_unmatched[key], dim=0)

    if args.save_histograms:
        n_rows = factors(args.num_parts)[-1]
        n_cols = factors(args.num_parts)[-2]
        # Combine the histograms
        plt.figure(figsize=(20, 20))
        for idx in range(args.num_parts):
            plt.subplot(n_rows, n_cols, idx + 1)
            plt.hist(part_logits_full_unmatched[idx], alpha=0.5, label=f"Part {idx} - Unmatched", density=True)
            plt.hist(part_logits_full_matched[idx], alpha=0.5, label=f"Part {idx} - Matched", density=True)
            plt.legend()
            plt.title(f"Part {idx}")
        plt.savefig(os.path.join(args.save_path, "part_logits_histogram.png"))
        plt.close()

    # Use the 5th percentile as the threshold
    for quantile in [0.01, 0.03, 0.05, 0.1]:
        thresholds = {}
        for key in part_logits_full_matched.keys():
            thresholds[key] = np.quantile(part_logits_full_matched[key].numpy(), quantile).item()
        # Remove threshold for bg (last part)
        thresholds.pop(args.num_parts)
        print(f"Quantile: {1 - quantile}")
        print(thresholds)
        save_json(thresholds, os.path.join(args.save_path, f"{1 - quantile}.json"))


if __name__ == '__main__':
    calc_part_logits(parse_args())
