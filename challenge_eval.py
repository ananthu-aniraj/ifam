# Ref: https://github.com/MadryLab/backgrounds_challenge/blob/master/challenge_eval.py
from torchvision import transforms
import torch as ch
import timm
import numpy as np
import json
import os
from argparse import ArgumentParser
from PIL import Image
import copy
from tqdm import tqdm
from tools_bg_challenge import ImageNet, ImageNet9, adv_bgs_eval_model, NormalizedModel
from load_model import load_model_2_stage
from utils.training_utils.engine_utils import load_state_dict_snapshot


def parse_args():
    parser = ArgumentParser()
    # Model
    parser.add_argument('--use_dinov2_baseline', default=False, action='store_true')
    parser.add_argument('--use_timm_baseline', default=False, action='store_true')
    parser.add_argument('--model_arch', default='resnet50', type=str,
                        help='pick model architecture')
    parser.add_argument('--use_hf_transformers', default=False, action='store_true')
    parser.add_argument('--num_parts', help='number of parts to predict',
                        default=8, type=int)
    parser.add_argument('--image_size', default=224, type=int)
    parser.add_argument('--output_stride', default=32, type=int)
    # Part Dropout
    parser.add_argument('--part_dropout', default=0.0, type=float)
    parser.add_argument('--part_dropout_stage_2', default=0.0, type=float)
    # Drop path
    parser.add_argument('--drop_path_stage_1', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--drop_path_stage_2', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    # Model Eps
    parser.add_argument('--eps', default=1e-10, type=float)
    # Gumbel Softmax
    parser.add_argument('--gumbel_softmax', default=False, action='store_true')
    parser.add_argument('--softmax_temperature', default=1.0, type=float)
    parser.add_argument('--gumbel_softmax_hard', default=False, action='store_true')
    # Model path
    parser.add_argument('--checkpoint', default=None, type=str)
    # Use soft masks for second stage
    parser.add_argument('--use_soft_masks', default=False, action='store_true')

    # Use part logit thresholds (only for evaluation)
    parser.add_argument('--part_logits_threshold_path', default="", type=str)
    # Data
    parser.add_argument('--class_mapping_file_path', default='in_to_in9.json', type=str, required=False)
    parser.add_argument('--batch_size', default=32, type=int, required=False)
    parser.add_argument('--num_workers', default=8, type=int, required=False)
    parser.add_argument('--data-path', required=True,
                        help='Path to the eval data')
    return parser.parse_args()


def main(args):
    args.eval_only = True
    args.pretrained_start_weights = False
    device = ch.device('cuda') if ch.cuda.is_available() else ch.device('cpu')
    map_to_in9 = {}
    with open(args.class_mapping_file_path, 'r') as f:
        map_to_in9.update(json.load(f))

    base_path_to_eval = args.data_path
    batch_size = args.batch_size
    workers = args.num_workers

    # Load model

    train_ds = ImageNet('/tmp')
    checkpoint = args.checkpoint
    # Load the model
    if args.use_dinov2_baseline:
        net = ch.hub.load('facebookresearch/dinov2', args.model_arch)
    elif args.use_timm_baseline:
        net = timm.create_model(args.model_arch, pretrained=True)
    else:
        net = load_model_2_stage(args, num_cls=1000)
        snapshot_data = ch.load(checkpoint, map_location=ch.device('cpu'), weights_only=True)
        if 'model_state' in snapshot_data:
            _, state_dict = load_state_dict_snapshot(snapshot_data)
        else:
            state_dict = copy.deepcopy(snapshot_data)
        net.load_state_dict(state_dict, strict=True)
    model = NormalizedModel(net, dataset=train_ds)
    model.to(device)
    model.eval()

    eval_baseline = args.use_dinov2_baseline or args.use_timm_baseline
    # Load backgrounds
    bg_ds = ImageNet9(f'{base_path_to_eval}/only_bg_t')
    bg_loader = bg_ds.make_loaders(batch_size=batch_size, workers=workers)

    # Load foregrounds
    fg_mask_base = f'{base_path_to_eval}/fg_mask/val'
    class_names = sorted(os.listdir(f'{fg_mask_base}'))

    def get_fgs(classnum):
        classname = class_names[classnum]
        return sorted(os.listdir(f'{fg_mask_base}/{classname}'))

    total_vulnerable = 0
    total_computed = 0
    per_class_challenge_accuracies = {}
    per_class_percent_vulnerable = {}
    # Big loop
    for fg_class in tqdm(range(9), desc='Classes'):

        fgs = get_fgs(fg_class)
        fg_classname = class_names[fg_class]
        total_vulnerable_per_class = 0
        total_computed_per_class = 0
        # Evaluate model
        for i in tqdm(range(len(fgs)), desc='Images', leave=False):
            if total_computed % 50 == 0:
                print(
                    f'At image {i} for class {fg_classname}')
                print(f'Up until now, have {total_vulnerable}/{total_computed} vulnerable foregrounds.')

            mask_name = fgs[i]
            fg_mask_path = f'{fg_mask_base}/{fg_classname}/{mask_name}'
            fg_mask = np.load(fg_mask_path)
            fg_mask = np.tile(fg_mask[:, :, np.newaxis], [1, 1, 3]).astype('uint8')
            fg_mask = transforms.ToTensor()(Image.fromarray(fg_mask * 255))

            img_name = mask_name.replace('npy', 'JPEG')
            image_path = f'{base_path_to_eval}/original/val/{fg_classname}/{img_name}'
            img = transforms.ToTensor()(Image.open(image_path))

            is_adv = adv_bgs_eval_model(bg_loader, model, img, fg_mask, fg_class, batch_size, map_to_in9,
                                        map_in_to_in9=True, device=device, eval_baseline=eval_baseline)
            # print(f'Image {i} of class {fg_classname} is {is_adv}.')
            total_vulnerable += is_adv
            total_computed += 1
            total_vulnerable_per_class += is_adv
            total_computed_per_class += 1
        percent_vulnerable_class = total_vulnerable_per_class / total_computed_per_class * 100
        per_class_challenge_accuracies[fg_classname] = 100 - percent_vulnerable_class
        per_class_percent_vulnerable[fg_classname] = percent_vulnerable_class
        print(f'Class {fg_classname} challenge accuracy: {100 - percent_vulnerable_class:.2f}%')
        print(f'Class {fg_classname} vulnerable foreground percentage: {percent_vulnerable_class:.2f}%')

    print('Evaluation complete')
    percent_vulnerable = total_vulnerable / total_computed * 100
    print(f'Summary: {total_vulnerable}/{total_computed} ({percent_vulnerable:.2f}%) are vulnerable foregrounds.')
    print(f'Summary: ({100 - percent_vulnerable:.2f}%) is the challenge accuracy.')
    print('Per class challenge accuracies:')
    for key, value in per_class_challenge_accuracies.items():
        print(f'{key}: {value:.2f}%')
    print('Per class vulnerable foreground percentages:')
    for key, value in per_class_percent_vulnerable.items():
        print(f'{key}: {value:.2f}%')


if __name__ == "__main__":
    args = parse_args()
    main(args)
