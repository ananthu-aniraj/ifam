"""
Attribute prediction loss (MAE)
"""

import torch
import torchvision


def mae_loss_attr(pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean"):
    """
    Calculate the mean absolute error between the predicted and target attributes
    :param pred: Predicted attributes
    :param target: Target attributes
    :param reduction: Reduction method
    :return: Mean absolute error
    """
    pred = torch.sigmoid(pred)
    loss = torch.nn.functional.l1_loss(pred, target, reduction=reduction)
    return loss


def bce_loss_attr(pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean"):
    """
    Calculate the binary cross-entropy loss between the predicted and target attributes
    :param pred: Predicted attributes
    :param target: Target attributes
    :param reduction: Reduction method
    :return: Binary cross-entropy loss
    """
    loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, target, reduction=reduction)
    return loss


def mil_loss_attr(pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean"):
    """
    Calculate the error between the predicted and target attributes using the Multiple Instance Learning (MIL) loss
    :param pred: Predicted attributes
    :param target: Target attributes
    :param reduction: Reduction method
    :return: Mean absolute error
    """
    # Split the target into two groups
    target_0 = target[target == 0]
    target_1 = target[target == 1]
    pred_0 = pred[target == 0]
    pred_1 = pred[target == 1]
    # Loss for targets = 0 (BCE)
    loss_0 = bce_loss_attr(pred_0, target_0, reduction=reduction)
    # Loss for targets = 1 (MAE)
    loss_1 = mae_loss_attr(pred_1, target_1, reduction=reduction)
    # Combine the losses
    loss = (loss_0 + loss_1) / 2
    return loss


def mse_loss_attr(pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean"):
    """
    Calculate the mean squared error between the predicted and target attributes
    :param pred: Predicted attributes
    :param target: Target attributes
    :param reduction: Reduction method
    :return: Mean squared error
    """
    pred = torch.sigmoid(pred)
    loss = torch.nn.functional.mse_loss(pred, target, reduction=reduction)
    return loss


def focal_loss_attr(pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean"):
    """
    Calculate the focal loss between the predicted and target attributes
    Args:
        pred: Predicted attributes
        target: Target attributes
        reduction: Reduction method

    Returns: The focal loss

    """
    loss = torchvision.ops.sigmoid_focal_loss(pred, target,
                                              reduction=reduction)  # Note: this function has an internal sigmoid
    return loss


class AttributePredictionLoss(torch.nn.Module):
    def __init__(self, loss_type: str = "mae", reduction: str = "mean"):
        super(AttributePredictionLoss, self).__init__()
        self.loss_type = loss_type
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        if self.loss_type == "mil":
            return mil_loss_attr(pred, target, self.reduction)
        elif self.loss_type == "mae":
            return mae_loss_attr(pred, target, self.reduction)
        elif self.loss_type == "bce":
            return bce_loss_attr(pred, target, self.reduction)
        elif self.loss_type == "mse":
            return mse_loss_attr(pred, target, self.reduction)
        elif self.loss_type == "focal":
            return focal_loss_attr(pred, target, self.reduction)
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")
