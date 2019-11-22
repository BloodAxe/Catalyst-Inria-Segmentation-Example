from pytorch_toolbelt.losses import *

__all__ = ["get_loss"]

from .dataset import UNLABELED_SAMPLE


def get_loss(loss_name: str, ignore_index=None):
    if loss_name.lower() == "bce":
        return SoftBCELoss(smooth_factor=0, ignore_index=ignore_index)

    if loss_name.lower() == "soft_bce":
        return SoftBCELoss(smooth_factor=0.1, ignore_index=ignore_index)

    if loss_name.lower() == "focal":
        return BinaryFocalLoss(alpha=None, gamma=1.5, ignore_index=ignore_index)

    raise KeyError(loss_name)
