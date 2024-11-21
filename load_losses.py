import torch
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import torch.nn.functional as F


class LabelSmoothingCrossEntropyNoReduction(LabelSmoothingCrossEntropy):
    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss


class SoftTargetCrossEntropyNoReduction(SoftTargetCrossEntropy):
    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss


def load_classification_loss(args, dataset_train, num_cls, reduction='mean'):
    """
    Load the loss function for classification
    :param args: Arguments from the argument parser
    :param dataset_train: Training dataset
    :param num_cls: Number of classes in the dataset
    :param reduction: Reduction method for the loss function
    :return:
    loss_fn: List of loss functions for training and evaluation
    """
    # Mixup/Cutmix
    mixup_fn = None
    mixup_active = args.turn_on_mixup_or_cutmix
    if mixup_active:
        print("Mixup is activated! Please note that this may not work with the equivariance loss")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=num_cls)

    # Define loss and optimizer
    if mixup_fn is not None:
        # smoothing is handled with mix-up label transform
        if reduction == 'mean':
            loss_fn_train = SoftTargetCrossEntropy()
        else:
            loss_fn_train = SoftTargetCrossEntropyNoReduction()
    elif args.smoothing > 0.:
        if reduction == 'mean':
            loss_fn_train = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            loss_fn_train = LabelSmoothingCrossEntropyNoReduction(smoothing=args.smoothing)
    else:
        loss_fn_train = torch.nn.CrossEntropyLoss(reduction=reduction)

    loss_fn_eval = torch.nn.CrossEntropyLoss(reduction=reduction)
    loss_fn = [loss_fn_train, loss_fn_eval]
    return loss_fn, mixup_fn


def load_loss_hyper_params(args):
    """
    Load the hyperparameters for the loss functions and affine transform parameters for equivariance
    :param args: Arguments from the argument parser
    :return:
    loss_hyperparams: Dictionary of loss hyperparameters
    eq_affine_transform_params: Dictionary of affine transform parameters for equivariance
    """
    loss_hyperparams = {'l_class_att': args.classification_loss, 'l_class_stage_2': args.classification_loss_stage_2,
                        'l_presence': args.presence_loss,
                        'l_presence_beta': args.presence_loss_beta, 'l_presence_type': args.presence_loss_type,
                        'l_equiv': args.equivariance_loss, 'l_conc': args.concentration_loss, 'l_tv': args.total_variation_loss,
                        'l_enforced_presence': args.enforced_presence_loss,
                        'l_pixel_wise_entropy': args.pixel_wise_entropy_loss,
                        'l_enforced_presence_loss_type': args.enforced_presence_loss_type,
                        'l_orth': args.orthogonality_loss_landmarks}

    # Affine transform parameters for equivariance
    degrees = [-args.degrees, args.degrees]
    translate = [args.translate_x, args.translate_y]
    scale = [args.scale_l, args.scale_u]
    shear_x = args.shear_x
    shear_y = args.shear_y
    shear = [shear_x, shear_y]
    if shear_x == 0.0 and shear_y == 0.0:
        shear = None

    eq_affine_transform_params = {'degrees': degrees, 'translate': translate, 'scale_ranges': scale, 'shear': shear}

    return loss_hyperparams, eq_affine_transform_params
