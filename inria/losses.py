from typing import Optional, Dict

import torch
import torch.nn.functional as F
from pytorch_toolbelt.losses import *
from torch import nn
from torch.nn import KLDivLoss

from inria.dataset import INPUT_MASK_KEY, INPUT_MASK_WEIGHT_KEY

__all__ = ["get_loss", "AdaptiveMaskLoss2d"]


class AdaptiveMaskLoss2d(nn.Module):
    """
    """

    def __init__(self, loss):
        super().__init__()
        self.loss = loss

    def forward(self, input, target):

        # Resize target to size of input
        target = F.interpolate(target, size=input.size()[2:], mode="bilinear", align_corners=False)
        target = target.clamp(0, 1)

        log_p = F.logsigmoid(input)
        loss = F.kl_div(log_p, target, reduction="mean")
        return loss


class WeightedBCEWithLogits(nn.Module):
    def __init__(self, mask_key, weight_key, ignore_index: Optional[int] = -100, reduction="mean"):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.weight_key = weight_key
        self.mask_key = mask_key

    def forward(self, label_input, target: Dict[str, torch.Tensor]):
        targets = target[self.mask_key]
        weights = target[self.weight_key]

        if self.ignore_index is not None:
            not_ignored_mask = (targets != self.ignore_index).float()

        loss = F.binary_cross_entropy_with_logits(label_input, targets, reduction="none") * weights

        if self.ignore_index is not None:
            loss = loss * not_ignored_mask.float()

        if self.reduction == "mean":
            loss = loss.mean()

        if self.reduction == "sum":
            loss = loss.sum()

        return loss


class KLDivLossWithLogits(KLDivLoss):
    """
    """

    def __init__(self):
        super().__init__()

    def forward(self, input, target):

        # Resize target to size of input
        target = F.interpolate(target, size=input.size()[2:], mode="bilinear", align_corners=False)

        input = torch.cat([input, 1 - input], dim=1)
        log_p = F.logsigmoid(input)

        target = torch.cat([target, 1 - target], dim=1)

        loss = F.kl_div(log_p, target, reduction="mean")
        return loss


def get_loss(loss_name: str, ignore_index=None):
    if loss_name.lower() == "kl":
        return KLDivLossWithLogits()

    if loss_name.lower() == "bce":
        return BCELoss(ignore_index=ignore_index)

    if loss_name.lower() == "wbce":
        return WeightedBCEWithLogits(
            mask_key=INPUT_MASK_KEY, weight_key=INPUT_MASK_WEIGHT_KEY, ignore_index=ignore_index
        )

    if loss_name.lower() == "soft_bce":
        return SoftBCELoss(smooth_factor=0.1, ignore_index=ignore_index)

    if loss_name.lower() == "focal":
        return BinaryFocalLoss(alpha=None, gamma=1.5, ignore_index=ignore_index)

    if loss_name.lower() == "jaccard":
        assert ignore_index is None
        return JaccardLoss(mode="binary")

    if loss_name.lower() == "lovasz":
        assert ignore_index is None
        return BinaryLovaszLoss()

    if loss_name.lower() == "log_jaccard":
        assert ignore_index is None
        return JaccardLoss(mode="binary", log_loss=True)

    if loss_name.lower() == "dice":
        assert ignore_index is None
        return DiceLoss(mode="binary", log_loss=False)

    if loss_name.lower() == "log_dice":
        assert ignore_index is None
        return DiceLoss(mode="binary", log_loss=True)

    raise KeyError(loss_name)
