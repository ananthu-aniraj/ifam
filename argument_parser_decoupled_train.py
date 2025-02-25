# Args Parser for training or evaluation (classification) of baseline models
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Model trainer for Image Classification'
    )
    # Data
    parser.add_argument('--data_path',
                        help='directory that contains cub files, must'
                             'contain folder "./images"', required=True)
    parser.add_argument('--image_sub_path_train', default='images',
                        help='subdirectory that contains training images')
    parser.add_argument('--image_sub_path_test', default='images',
                        help='subdirectory that contains test images')
    parser.add_argument('--dataset', default='cub', type=str)
    parser.add_argument('--train_split', default=0.9, type=float, help='fraction of training data to use')
    parser.add_argument('--eval_mode', default='val', type=str,
                        help='which split to use for evaluation')
    parser.add_argument('--anno_path_train', default='', type=str, required=False)
    parser.add_argument('--anno_path_test', default='', type=str, required=False)
    parser.add_argument('--metadata_path', default='', type=str, required=False)
    parser.add_argument('--species_id_to_name_file', default='', type=str, required=False)
    parser.add_argument('--mask_sub_path_train', default='segmentations', type=str, required=False)
    parser.add_argument('--mask_sub_path_test', default='segmentations', type=str, required=False)

    # Training
    parser.add_argument('--snapshot_dir', type=str)
    parser.add_argument('--save_every_n_epochs', default=10, type=int)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--scratch_lr_factor', default=100.0, type=float)
    parser.add_argument('--epochs', type=int, default=28)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--grad_accumulation_steps', default=1, type=int)

    # * Misc training params
    parser.add_argument('--grad_norm_clip', default=0.0, type=float)
    parser.add_argument('--use_amp', action='store_true', default=False)

    # Evaluation params
    parser.add_argument('--eval_only', default=False, action='store_true',
                        help='Whether to only eval the model')
    parser.add_argument('--crop_pct', type=float, default=None)

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Augmentation parameters
    parser.add_argument('--image_size', default=448, type=int)

    # Model params
    parser.add_argument('--model_arch', default='resnet50', type=str,
                        help='pick model architecture')
    parser.add_argument('--use_hf_transformers', default=False, action='store_true')
    parser.add_argument('--use_torchvision_resnet_model', default=False, action='store_true')
    parser.add_argument('--pretrained_start_weights', default=False, action='store_true')
    parser.add_argument('--pretrained_model_path', default=None, type=str)
    parser.add_argument('--drop_path', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--output_stride', type=int, default=32, help='stride of the model')
    parser.add_argument('--freeze_backbone', default=False, action='store_true')
    parser.add_argument('--pooling_type', default="cls", type=str)

    # * Optimizer params
    parser.add_argument('--optimizer_type', default='adam', type=str)
    parser.add_argument('--weight_decay', default=0.05, type=float, help='normalized weight decay')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--betas1', default=0.9, type=float)
    parser.add_argument('--betas2', default=0.999, type=float)
    parser.add_argument('--dampening', default=0.0, type=float)
    parser.add_argument('--trust_coeff', default=0.001, type=float)
    parser.add_argument('--always_adapt', action='store_true', default=False)
    parser.add_argument('--turn_off_grad_averaging', action='store_true', default=False)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--use_zero', default=False, action='store_true')

    # * Scheduler params
    parser.add_argument('--scheduler_type', default='cosine',
                        choices=['cosine', 'linearlr', 'steplr'],
                        type=str)
    parser.add_argument('--scheduler_warmup_epochs', default=0, type=int)
    parser.add_argument('--warmup_lr', type=float, default=0.0)
    parser.add_argument('--scheduler_restart_factor', default=1, type=int)
    parser.add_argument('--scheduler_gamma', default=0.1, type=float)
    parser.add_argument('--scheduler_step_size', default=10, type=int)
    parser.add_argument('--min_lr', type=float, default=0.0, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--cosine_cycle_limit', default=1, type=int)

    # Wandb params
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project', default='fine-tune-cnn', type=str)
    parser.add_argument('--job_type', default='fine_tune_dino_v2', type=str)
    parser.add_argument('--log_interval', default=10, type=int)
    parser.add_argument('--group', default='vit_base', type=str)
    parser.add_argument('--wandb_entity', default='', type=str)
    parser.add_argument('--wandb_mode', default='online', type=str, choices=['online', 'offline'])

    # * Resume training params
    parser.add_argument('--resume_training', action='store_true', default=False)
    parser.add_argument('--wandb_resume_id', default=None, type=str)

    # Array training job
    parser.add_argument('--array_training_job', default=False, action='store_true',
                        help='Whether to run as an array job (i.e. training with multiple random seeds on the same settings)')

    # Model Iterate Averaging Type
    group = parser.add_argument_group('Model iterate average parameters')
    group.add_argument('--averaging_type', default='', type=str, help='Type of model iterate averaging to use')
    group.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                       help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
    group.add_argument('--model-ema-decay', type=float, default=0.9999,
                       help='Decay factor for model weights moving average (default in timm: 0.9999)')
    group.add_argument('--no-model-ema-warmup', action='store_true',
                       help='Enable warmup for model EMA decay.')

    # Use late masking
    parser.add_argument('--late_masking', default=False, action='store_true')

    # BCE Loss (for multi-class classification) from timm
    parser.add_argument('--use_bce_loss', default=False, action='store_true')
    parser.add_argument('--bce-sum', action='store_true', default=False,
                        help='Sum over classes when using BCE loss.')
    parser.add_argument('--bce-target-thresh', type=float, default=None,
                        help='Threshold for binarizing softened BCE targets (default: None, disabled).')
    parser.add_argument('--bce-pos-weight', type=float, default=None,
                        help='Positive weighting for BCE loss.')
    # Label Smoothing
    parser.add_argument('--smoothing', type=float, default=0.0,
                        help='Label smoothing (default: 0.0)')
    args = parser.parse_args()
    return args
