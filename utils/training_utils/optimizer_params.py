import torch
import math
import inspect
from timm.optim.lars import Lars
from timm.optim.lamb import Lamb
from utils.training_utils.ddp_utils import calculate_effective_batch_size


def build_optimizer(args, params_groups, weight_decay):
    """
    Function to build the optimizer
    :param args: arguments from the command line
    :param params_groups: parameters to be optimized
    :param weight_decay: weight decay
    :return: optimizer
    """
    grad_averaging = not args.turn_off_grad_averaging
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.optimizer_type == 'adamw':
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        return torch.optim.AdamW(params=params_groups, betas=(args.betas1, args.betas2), lr=args.lr,
                                 weight_decay=weight_decay, **extra_args)
    elif args.optimizer_type == 'sgd':
        fused_available = 'fused' in inspect.signature(torch.optim.SGD).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        return torch.optim.SGD(params=params_groups, lr=args.lr, momentum=args.momentum, weight_decay=weight_decay,
                               nesterov=True, **extra_args)
    elif args.optimizer_type == 'adam':
        fused_available = 'fused' in inspect.signature(torch.optim.Adam).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        return torch.optim.Adam(params=params_groups, betas=(args.betas1, args.betas2), lr=args.lr,
                                weight_decay=weight_decay, **extra_args)
    elif args.optimizer_type == 'nadam':
        fused_available = 'fused' in inspect.signature(torch.optim.NAdam).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        return torch.optim.NAdam(params=params_groups, betas=(args.betas1, args.betas2), lr=args.lr,
                                 weight_decay=weight_decay, **extra_args)
    elif args.optimizer_type == 'lars':
        return Lars(params=params_groups, lr=args.lr, momentum=args.momentum, weight_decay=weight_decay,
                    dampening=args.dampening, trust_coeff=args.trust_coeff, trust_clip=False,
                    always_adapt=args.always_adapt)
    elif args.optimizer_type == 'nlars':
        return Lars(params=params_groups, lr=args.lr, momentum=args.momentum, weight_decay=weight_decay,
                    dampening=args.dampening, nesterov=True, trust_coeff=args.trust_coeff, trust_clip=False,
                    always_adapt=args.always_adapt)
    elif args.optimizer_type == 'larc':
        return Lars(params=params_groups, lr=args.lr, momentum=args.momentum, weight_decay=weight_decay,
                    dampening=args.dampening, trust_coeff=args.trust_coeff, trust_clip=True,
                    always_adapt=args.always_adapt)
    elif args.optimizer_type == 'nlarc':
        return Lars(params=params_groups, lr=args.lr, momentum=args.momentum, weight_decay=weight_decay,
                    dampening=args.dampening, nesterov=True, trust_coeff=args.trust_coeff, trust_clip=True,
                    always_adapt=args.always_adapt)
    elif args.optimizer_type == 'lamb':
        return Lamb(params=params_groups, lr=args.lr, betas=(args.betas1, args.betas2), weight_decay=weight_decay,
                    grad_averaging=grad_averaging, max_grad_norm=args.max_grad_norm, trust_clip=False,
                    always_adapt=args.always_adapt)
    elif args.optimizer_type == 'lambc':
        return Lamb(params=params_groups, lr=args.lr, betas=(args.betas1, args.betas2), weight_decay=weight_decay,
                    grad_averaging=grad_averaging, max_grad_norm=args.max_grad_norm, trust_clip=True,
                    always_adapt=args.always_adapt)
    else:
        raise NotImplementedError(f'Optimizer {args.optimizer_type} not implemented.')


def calculate_weight_decay(args, dataset_train):
    """
    Function to calculate the weight decay
    Implementation of normalized weight decay as per the paper "Decoupled Weight Decay Regularization": https://arxiv.org/pdf/1711.05101.pdf
    :param args: Arguments from the command line
    :param dataset_train: Training dataset
    :return: weight_decay: Weight decay
    """
    batch_size = calculate_effective_batch_size(args)
    num_iterations = len(dataset_train) // batch_size  # Since we set drop_last=True
    norm_weight_decay = args.weight_decay
    weight_decay = norm_weight_decay * math.sqrt(1 / (num_iterations * args.epochs))
    return weight_decay


def layer_group_matcher_baseline(args, model, dataset_train):
    """
    Function to group the parameters of the model into different groups
    :param args: Arguments from the command line
    :param model: Model to be trained
    :param dataset_train: Training dataset
    :return: param_groups: Parameters grouped into different groups
    """
    weight_decay = calculate_weight_decay(args, dataset_train)
    scratch_layers = ["head.", "fc.", "attn_pool.", "sim_pool", "fc_norm"]
    scratch_parameters = []
    no_weight_decay_params_scratch = []
    finetune_parameters = []
    no_weight_decay_params_bb = []
    for name, p in model.named_parameters():

        if any(x in name for x in scratch_layers):
            print("scratch layer_name: " + name)
            if p.ndim == 1:
                no_weight_decay_params_scratch.append(p)
            else:
                scratch_parameters.append(p)
        else:
            if p.ndim == 1:
                no_weight_decay_params_bb.append(p)
            else:
                finetune_parameters.append(p)
            if args.freeze_backbone:
                p.requires_grad = False
            else:
                p.requires_grad = True

    param_groups = [{'params': finetune_parameters, 'lr': args.lr},
                    {'params': no_weight_decay_params_bb, 'lr': args.lr, 'weight_decay': 0.0},
                    {'params': scratch_parameters, 'lr': args.lr * args.scratch_lr_factor}]
    if len(no_weight_decay_params_scratch) > 0:
        param_groups.append(
            {'params': no_weight_decay_params_scratch, 'lr': args.lr * args.scratch_lr_factor, 'weight_decay': 0.0})
    return param_groups, weight_decay


def layer_group_matcher_pdisco_2_stage(args, model, dataset_train):
    """
    Function to group the parameters of the model into different groups
    :param args: Arguments from the command line
    :param model: Model to be trained
    :param dataset_train: Training dataset
    :return: param_groups: Parameters grouped into different groups
    """
    weight_decay = calculate_weight_decay(args, dataset_train)
    scratch_layers = ["fc_class_landmarks"]
    scratch_layers_stage_2 = ["head", "head_norm", "fc_norm"]
    modulation_layers = ["modulation"]
    finer_layers = ["fc_landmarks"]
    finer_layers_no_wd = ["constant_tensor", "softmax_temperature", "landmark_norm"]
    if args.use_hf_transformers:
        unfrozen_layers_stage_1 = ["stage_1.embeddings.cls_token", "stage_1.embeddings.position_embeddings",
                                   "stage_1.embeddings.reg_token"]
        unfrozen_layers_stage_2 = ["stage_2.embeddings.cls_token", "stage_2.embeddings.position_embeddings",
                                   "stage_2.embeddings.reg_token", "stage_2.embeddings.part_embed"]
    else:
        unfrozen_layers_stage_1 = ["stage_1.cls_token", "stage_1.pos_embed", "stage_1.reg_token"]
        unfrozen_layers_stage_2 = ["stage_2.cls_token", "stage_2.pos_embed", "stage_2.reg_token", "stage_2.part_embed"]
    stage_2_layers = ["stage_2"]
    scratch_parameters = []
    scratch_parameters_no_wd = []
    scratch_parameters_stage_2 = []
    scratch_parameters_stage_2_no_wd = []
    modulation_parameters = []
    stage_2_parameters_wd = []
    stage_2_parameters_no_wd = []
    backbone_parameters_wd = []
    no_weight_decay_params = []
    no_weight_decay_params_stage_2 = []
    finer_parameters = []
    finer_parameters_no_wd = []

    for name, p in model.named_parameters():
        if any(x in name for x in scratch_layers):
            # print("scratch layer_name: " + name)
            if p.ndim == 1:
                scratch_parameters_no_wd.append(p)
            else:
                scratch_parameters.append(p)
            p.requires_grad = True

        elif any(x in name for x in scratch_layers_stage_2):
            # print("scratch layer_name: " + name)
            if p.ndim == 1:
                scratch_parameters_stage_2_no_wd.append(p)
            else:
                scratch_parameters_stage_2.append(p)
            p.requires_grad = True

        elif any(x in name for x in modulation_layers):
            # print("modulation layer_name: " + name)
            modulation_parameters.append(p)
            p.requires_grad = True

        elif any(x in name for x in finer_layers):
            # print("finer layer_name: " + name)
            finer_parameters.append(p)
            p.requires_grad = True

        elif any(x in name for x in finer_layers_no_wd):
            # print("finer layer_name: " + name)
            finer_parameters_no_wd.append(p)
            p.requires_grad = True

        elif any(x in name for x in unfrozen_layers_stage_1):
            # print("unfrozen layer stage_1: " + name)
            no_weight_decay_params.append(p)
            if args.freeze_params_stage_1:
                p.requires_grad = False
            else:
                p.requires_grad = True

        elif any(x in name for x in unfrozen_layers_stage_2):
            # print("unfrozen layer stage_2: " + name)
            no_weight_decay_params_stage_2.append(p)
            if args.freeze_params_stage_2:
                p.requires_grad = False
            else:
                p.requires_grad = True

        elif any(x in name for x in stage_2_layers):
            # print("stage_2 layer_name: " + name)
            if p.ndim == 1:
                stage_2_parameters_no_wd.append(p)
            else:
                stage_2_parameters_wd.append(p)
            if args.freeze_second_stage:
                p.requires_grad = False
            else:
                p.requires_grad = True

        else:
            if args.freeze_backbone:
                p.requires_grad = False
            else:
                p.requires_grad = True

            if p.ndim == 1:
                no_weight_decay_params.append(p)
            else:
                backbone_parameters_wd.append(p)

    param_groups = [{'params': backbone_parameters_wd, 'lr': args.lr},
                    {'params': no_weight_decay_params, 'lr': args.lr, 'weight_decay': 0.0},
                    {'params': no_weight_decay_params_stage_2, 'lr': args.lr * args.stage_two_lr_factor,
                     'weight_decay': 0.0},
                    {'params': stage_2_parameters_wd, 'lr': args.lr * args.stage_two_lr_factor},
                    {'params': stage_2_parameters_no_wd, 'lr': args.lr * args.stage_two_lr_factor, 'weight_decay': 0.0},
                    {'params': finer_parameters, 'lr': args.lr * args.finer_lr_factor, 'weight_decay': 0.0},
                    {'params': finer_parameters_no_wd, 'lr': args.lr * args.finer_lr_factor, 'weight_decay': 0.0},
                    {'params': modulation_parameters, 'lr': args.lr * args.modulation_lr_factor, 'weight_decay': 0.0},
                    {'params': scratch_parameters, 'lr': args.lr * args.scratch_lr_factor},
                    {'params': scratch_parameters_no_wd, 'lr': args.lr * args.scratch_lr_factor, 'weight_decay': 0.0},
                    {'params': scratch_parameters_stage_2, 'lr': args.lr * args.scratch_lr_factor},
                    {'params': scratch_parameters_stage_2_no_wd, 'lr': args.lr * args.scratch_lr_factor, 'weight_decay': 0.0}]

    return param_groups, weight_decay
