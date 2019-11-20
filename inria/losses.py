from pytorch_toolbelt.losses import *
from torch.nn import BCEWithLogitsLoss

__all__ = ["get_loss"]


def get_loss(loss_name: str, **kwargs):
    if loss_name.lower() == "bce":
        return BCEWithLogitsLoss(**kwargs)

    if loss_name.lower() == "soft_bce":
        return SoftBCELoss(smooth_factor=1e-5, ignore_index=127)

    if loss_name.lower() == "focal":
        return BinaryFocalLoss(alpha=None, gamma=1.5, **kwargs)

    raise KeyError(loss_name)
