from torch import nn

from . import fpn, unet, deeplab

__all__ = ["get_model"]


def get_model(model_name: str, dropout=0.0) -> nn.Module:
    registry = {
        # FPN family using concatentation
        "resnet34_fpncat128": fpn.resnet34_fpncat128,
        "resnet152_fpncat256": fpn.resnet152_fpncat256,
        "seresnext50_fpncat128": fpn.seresnext50_fpncat128,
        "effnetb4_fpncat128": fpn.effnetB4_fpncat128,
        "seresnext101_fpncat256": fpn.seresnext101_fpncat256,

        # FPN family using summation
        "seresnext101_fpnsum256": fpn.seresnext101_fpnsum256,

        # RFPN family
        "resnet34_rfpncat128": fpn.resnet34_rfpncat128,
        "seresnext101_rfpncat256": fpn.seresnext101_rfpncat256,

        # UNet
        "resnet34_unet32": unet.resnet34_unet32,
        "resnet34_unet32v2": unet.resnet34_unet32v2,
        "seresnext50_unet64": unet.seresnext50_unet64,
        "seresnext101_unet64": unet.seresnext101_unet64,

        # Deeplab
        "resnet34_deeplab128": deeplab.resnet34_deeplab128,
        "seresnext101_deeplab256": deeplab.seresnext101_deeplab256
    }

    return registry[model_name.lower()](dropout=dropout)
