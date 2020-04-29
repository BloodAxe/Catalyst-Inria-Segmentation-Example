from typing import Optional, Dict

import torch
import torch.nn.functional as F
from kornia import get_gaussian_kernel2d, filter2D
from kornia.losses import SSIM
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


class SSIMWithLogits(nn.Module):
    def __init__(self, window_size: int, max_val=1.0, reduction="mean"):
        super().__init__()
        self.window_size: int = window_size
        self.max_val: float = max_val
        self.reduction: str = reduction

        self.window: torch.Tensor = get_gaussian_kernel2d((window_size, window_size), (1.5, 1.5))
        self.window = self.window.requires_grad_(False)  # need to disable gradients

        self.padding: int = (window_size - 1) // 2

        self.C1: float = (0.01 * self.max_val) ** 2
        self.C2: float = (0.03 * self.max_val) ** 2

    def forward(self, input, target):
        p = F.logsigmoid(input).exp()
        return self.compute(p, target)

    def compute(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:

        if not torch.is_tensor(img1):
            raise TypeError("Input img1 type is not a torch.Tensor. Got {}".format(type(img1)))

        if not torch.is_tensor(img2):
            raise TypeError("Input img2 type is not a torch.Tensor. Got {}".format(type(img2)))

        if not len(img1.shape) == 4:
            raise ValueError("Invalid img1 shape, we expect BxCxHxW. Got: {}".format(img1.shape))

        if not len(img2.shape) == 4:
            raise ValueError("Invalid img2 shape, we expect BxCxHxW. Got: {}".format(img2.shape))

        if not img1.shape == img2.shape:
            raise ValueError("img1 and img2 shapes must be the same. Got: {}".format(img1.shape, img2.shape))

        if not img1.device == img2.device:
            raise ValueError("img1 and img2 must be in the same device. Got: {}".format(img1.device, img2.device))

        if not img1.dtype == img2.dtype:
            raise ValueError("img1 and img2 must be in the same dtype. Got: {}".format(img1.dtype, img2.dtype))

            # prepare kernel
        b, c, h, w = img1.shape
        tmp_kernel: torch.Tensor = self.window.to(img1.device).to(img1.dtype)
        tmp_kernel = torch.unsqueeze(tmp_kernel, dim=0)

        # compute local mean per channel
        mu1: torch.Tensor = filter2D(img1, tmp_kernel)
        mu2: torch.Tensor = filter2D(img2, tmp_kernel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # compute local sigma per channel
        sigma1_sq = filter2D(img1 * img1, tmp_kernel) - mu1_sq
        sigma2_sq = filter2D(img2 * img2, tmp_kernel) - mu2_sq
        sigma12 = filter2D(img1 * img2, tmp_kernel) - mu1_mu2

        ssim_map = ((2.0 * mu1_mu2 + self.C1) * (2.0 * sigma12 + self.C2)) / (
            (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)
        )

        loss = torch.clamp(-ssim_map + 1.0, min=0, max=1) / 2.0

        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "none":
            pass
        return loss


def get_loss(loss_name: str, ignore_index=None):
    if loss_name.lower() == "kl":
        return KLDivLossWithLogits()

    if loss_name.lower() == "bce":
        return SoftBCEWithLogitsLoss(ignore_index=ignore_index)

    if loss_name.lower() == "wbce":
        return WeightedBCEWithLogits(
            mask_key=INPUT_MASK_KEY, weight_key=INPUT_MASK_WEIGHT_KEY, ignore_index=ignore_index
        )

    if loss_name.lower() == "soft_bce":
        return SoftBCEWithLogitsLoss(smooth_factor=0.1, ignore_index=ignore_index)

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

    if loss_name.lower() == "ssim":
        assert ignore_index is None
        return SSIMWithLogits(7).cuda()

    raise KeyError(loss_name)
