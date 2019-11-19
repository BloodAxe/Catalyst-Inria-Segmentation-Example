from torch import nn

from . import fpn, unet

__all__ = ["get_model"]


def get_model(model_name: str, dropout=0.0) -> nn.Module:
    registry = {
        # FPN family
        "resnet34_fpncat128": fpn.resnet34_fpncat128,
        "resnet152_fpncat256": fpn.resnet152_fpncat256,
        "seresnext50_fpncat128": fpn.seresnext50_fpncat128,
        "effnetB4_fpncat128": fpn.effnetB4_fpncat128,
        "seresnext101_fpncat256": fpn.seresnext101_fpncat256,

        # UNet
        "seresnext101_unet64": unet.seresnext101_unet64
    }

    return registry[model_name.lower()](dropout=dropout)
