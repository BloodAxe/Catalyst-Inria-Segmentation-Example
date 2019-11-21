from pytorch_toolbelt.losses import *
from torch.nn import BCEWithLogitsLoss

__all__ = ["get_loss"]

from .dataset import UNLABELED_SAMPLE


def get_loss(loss_name: str, **kwargs):
    if loss_name.lower() == "bce":
        return BCEWithLogitsLoss(**kwargs)

    if loss_name.lower() == "soft_bce":
        return SoftBCELoss(smooth_factor=0, ignore_index=UNLABELED_SAMPLE)

    if loss_name.lower() == "focal":
        return BinaryFocalLoss(alpha=None, gamma=1.5, **kwargs)

    raise KeyError(loss_name)
