# Args Parser for training or evaluation (classification) of models
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Model trainer for Image Classification'
    )
    parser.add_argument('--model_arch', default='hf_hub:timm/vit_base_patch14_reg4_dinov2.lvd142m', type=str,
                        help='pick model architecture')
    parser.add_argument('--use_hf_transformers', default=False, action='store_true')
    parser.add_argument('--use_torchvision_resnet_model', default=False, action='store_true')

    # Data
    parser.add_argument('--data_path',
                        help='directory that contains cub files', required=True)
    parser.add_argument('--image_sub_path_train', default='images',
                        help='subdirectory that contains training images')
    parser.add_argument('--image_sub_path_test', default='images',
                        help='subdirectory that contains test images')
    parser.add_argument('--dataset', default='cub', type=str)
    parser.add_argument('--train_split', default=0.9, type=float, help='fraction of training data to use')
    parser.add_argument('--eval_mode', default='val', type=str,
                        help='which split to use for evaluation')
    parser.add_argument('--mask_sub_path', default='all_masks', type=str, required=False)

    # Training
    parser.add_argument('--snapshot_dir', type=str)
    parser.add_argument('--save_every_n_epochs', default=10, type=int)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=28)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--grad_accumulation_steps', default=1, type=int)

    # Attention map saving probability
    parser.add_argument('--amap_saving_prob', default=0.05, type=float)

    # * Misc training params
    parser.add_argument('--grad_norm_clip', default=2.0, type=float)
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
    parser.add_argument('--augmentations_to_use', type=str, default='cub_original',
                        choices=['timm', 'cub_original', 'siim_acr'])
    parser.add_argument('--image_size', default=448, type=int)
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m6-mstd0.5', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m6-mstd0.5)'),
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--imagenet_default_mean_and_std', action='store_false', default=True)
    parser.add_argument('--hflip', type=float, default=0.5, help='Horizontal flip probability')
    parser.add_argument('--vflip', type=float, default=0., help='Vertical flip probability')

    # Random Erase params
    parser.add_argument('--reprob', type=float, default=0.0, metavar='PCT',
                        help='Random erase prob (default: 0.0)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # Model params
    parser.add_argument('--num_parts', help='number of parts to predict',
                        default=8, type=int)
    parser.add_argument('--pretrained_start_weights', default=False, action='store_true')
    parser.add_argument('--pretrained_model_path', default=None, type=str)
    parser.add_argument('--drop_path_stage_1', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--drop_path_stage_2', type=float, default=0, metavar='PCT', help='Drop path rate (default: 0.0)')
    parser.add_argument('--output_stride', type=int, default=32, help='stride of the model')
    parser.add_argument('--freeze_backbone', default=False, action='store_true')
    parser.add_argument('--freeze_params_stage_1', default=False, action='store_true')
    parser.add_argument('--freeze_params_stage_2', default=False, action='store_true')
    parser.add_argument('--freeze_second_stage', default=False, action='store_true')

    # * Optimizer params
    parser.add_argument('--optimizer_type', default='adam', type=str)
    parser.add_argument('--weight_decay', default=0, type=float, help='normalized weight decay')
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
    parser.add_argument('--warmup_lr', type=float, default=0)
    parser.add_argument('--scheduler_restart_factor', default=1, type=int)
    parser.add_argument('--scheduler_gamma', default=0.1, type=float)
    parser.add_argument('--scheduler_step_size', default=10, type=int)
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--cosine_cycle_limit', default=1, type=int)

    # * LR params for each param group
    parser.add_argument('--lr', default=1e-6, type=float)
    parser.add_argument('--scratch_lr_factor', default=1e4, type=float)
    parser.add_argument('--finer_lr_factor', default=1e3, type=float)
    parser.add_argument('--modulation_lr_factor', default=1e4, type=float)
    parser.add_argument('--stage_two_lr_factor', default=1, type=float)

    # Wandb params
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project', default='', type=str)
    parser.add_argument('--job_type', default='', type=str)
    parser.add_argument('--log_interval', default=10, type=int)
    parser.add_argument('--group', default='vit_base', type=str)
    parser.add_argument('--wandb_entity', default='', type=str)
    parser.add_argument('--wandb_mode', default='online', type=str, choices=['online', 'offline'])

    # * Resume training params
    parser.add_argument('--resume_training', action='store_true', default=False)
    parser.add_argument('--wandb_resume_id', default=None, type=str)

    # Loss hyperparameters
    parser.add_argument('--classification_loss', default=1.0, type=float)
    parser.add_argument('--classification_loss_stage_2', default=1.0, type=float)
    parser.add_argument('--presence_loss', default=1.0, type=float)
    parser.add_argument('--presence_loss_beta', default=0.1, type=float)
    parser.add_argument('--presence_loss_type', default="original",
                        choices=["original", "soft_constraint", "tanh", "soft_tanh"], type=str)
    parser.add_argument('--concentration_loss', default=0, type=float)
    parser.add_argument('--equivariance_loss', default=1.0, type=float)
    parser.add_argument('--total_variation_loss', default=1.0, type=float)
    parser.add_argument('--enforced_presence_loss', default=2.0, type=float)
    parser.add_argument('--enforced_presence_loss_type', default="enforced_presence", choices=["linear", "log", "mse", "enforced_presence"],
                        type=str)
    parser.add_argument('--pixel_wise_entropy_loss', default=1.0, type=float)
    parser.add_argument('--orthogonality_loss_landmarks', default=1.0, type=float)

    # Equivariance affine transform params
    parser.add_argument('--degrees', default=90, type=float)
    parser.add_argument('--translate_x', default=0.11, type=float)
    parser.add_argument('--translate_y', default=0.11, type=float)
    parser.add_argument('--scale_l', default=0.8, type=float)
    parser.add_argument('--scale_u', default=1.4, type=float)
    parser.add_argument('--shear_x', default=0.0, type=float)
    parser.add_argument('--shear_y', default=0.0, type=float)

    # Part Dropout
    parser.add_argument('--part_dropout', default=0.3, type=float)
    parser.add_argument('--part_dropout_stage_2', default=0.3, type=float)

    # Gumbel Softmax
    parser.add_argument('--gumbel_softmax', default=False, action='store_true')
    parser.add_argument('-softmax_temperature', default=1.0, type=float)
    parser.add_argument('--gumbel_softmax_hard', default=False, action='store_true')

    # Array training job
    parser.add_argument('--array_training_job', default=False, action='store_true',
                        help='Whether to run as an array job (i.e. training with multiple random seeds on the same settings)')

    # Use part logit thresholds (only for evaluation)
    parser.add_argument('--part_logits_threshold_path', default="", type=str)

    # Intervention parameters
    parser.add_argument('--intervention_fixed_part_id', nargs="*", type=int, default=[])

    # Use soft masks for second stage
    parser.add_argument('--use_soft_masks', default=False, action='store_true')

    # Label Smoothing
    parser.add_argument('--smoothing', type=float, default=0.0,
                        help='Label smoothing (default: 0.0)')

    # Model Iterate Averaging Type
    group = parser.add_argument_group('Model iterate average parameters')
    group.add_argument('--averaging_type', default='', type=str, help='Type of model iterate averaging to use')
    group.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                       help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
    group.add_argument('--model-ema-decay', type=float, default=0.9999,
                       help='Decay factor for model weights moving average (default in timm: 0.9999)')
    group.add_argument('--no-model-ema-warmup', action='store_true',
                       help='Enable warmup for model EMA decay.')

    args = parser.parse_args()
    return args
