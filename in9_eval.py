# Ref: https://github.com/MadryLab/backgrounds_challenge/blob/master/in9_eval.py
from torchvision import transforms
from torchvision.models import get_model
import torch as ch
import timm
import json
import os
import copy
from argparse import ArgumentParser
from tools_bg_challenge import ImageNet, ImageNet9, eval_model, NormalizedModel
from load_model import load_model_2_stage
from utils.training_utils.engine_utils import load_state_dict_snapshot


def parse_args():
    parser = ArgumentParser()
    # Model
    parser.add_argument('--use_dinov2_baseline', default=False, action='store_true')
    parser.add_argument('--use_timm_baseline', default=False, action='store_true')
    parser.add_argument('--use_torchvision_baseline', default=False, action='store_true')
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
    parser.add_argument('--eval-dataset', default='original',
                        help='What IN-9 variation to evaluate on.')
    args = parser.parse_args()
    return args


def main(args):
    args.eval_only = True
    args.pretrained_start_weights = False
    device = ch.device("cuda:0" if ch.cuda.is_available() else "cpu")
    map_to_in9 = {}
    with open(args.class_mapping_file_path, 'r') as f:
        map_to_in9.update(json.load(f))

    base_path_to_eval = args.data_path
    batch_size = args.batch_size
    workers = args.num_workers

    # Load eval dataset
    variation = args.eval_dataset
    in9_ds = ImageNet9(os.path.join(base_path_to_eval, variation))
    val_loader = in9_ds.make_loaders(batch_size=batch_size, workers=workers)

    # Load model
    train_ds = ImageNet('/tmp')
    checkpoint = args.checkpoint

    # Load the model
    if args.use_dinov2_baseline:
        net = ch.hub.load('facebookresearch/dinov2', args.model_arch)
    elif args.use_torchvision_baseline:
        net = get_model(args.model_arch, weights="DEFAULT")
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
    eval_baseline = args.use_dinov2_baseline or args.use_timm_baseline or args.use_torchvision_baseline
    model = NormalizedModel(net, dataset=train_ds)
    model.to(device)
    model.eval()

    # Evaluate model
    prec1 = eval_model(val_loader, model, map_to_in9, map_in_to_in9=True, device=device, eval_baseline=eval_baseline)
    print(f'Accuracy on {variation} is {prec1 * 100:.2f}%')


if __name__ == "__main__":
    args = parse_args()
    main(args)
