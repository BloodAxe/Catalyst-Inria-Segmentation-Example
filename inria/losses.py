import torch
from pytorch_toolbelt.losses import *
import torch.nn.functional as F

__all__ = ["get_loss", "AdaptiveMaskLoss2d"]

from torch import nn


def get_loss(loss_name: str, ignore_index=None):
    if loss_name.lower() == "bce":
        return BCELoss(ignore_index=ignore_index)

    if loss_name.lower() == "soft_bce":
        return SoftBCELoss(smooth_factor=0.1, ignore_index=ignore_index)

    if loss_name.lower() == "focal":
        return BinaryFocalLoss(alpha=None, gamma=1.5, ignore_index=ignore_index)

    if loss_name.lower() == "jaccard":
        assert ignore_index is None
        return JaccardLoss(mode="binary")

    raise KeyError(loss_name)


class AdaptiveMaskLoss2d(nn.Module):
    """
    Works only with sigmoid masks and bce loss
    Rescales target mask to predicted mask
    """

    def __init__(self, loss):
        super().__init__()
        self.loss = loss

    def forward(self, input, target):

        # Resize target to size of input
        input = F.interpolate(
            input, size=input.size()[2:], mode="bilinear", align_corners=False
        )

        return self.loss(input, target)
