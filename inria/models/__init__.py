from torch import nn

from . import fpn, unet, deeplab, hrnet, runet, hg

__all__ = ["get_model"]

from .unet import sexresnet50_unet128, xresnet50_unet128
from .unet_hm import densenet121_unet64_offset


def get_model(model_name: str, dropout=0.0, pretrained=True) -> nn.Module:
    registry = {
        # FPN family using concatenation
        "resnet34_fpncat128": fpn.resnet34_fpncat128,
        "resnet152_fpncat256": fpn.resnet152_fpncat256,
        "seresnext50_fpncat128": fpn.seresnext50_fpncat128,
        "seresnext101_fpncat256": fpn.seresnext101_fpncat256,
        # FPN family using summation
        "seresnext101_fpnsum256": fpn.seresnext101_fpnsum256,
        # UNet
        "seresnext50_unet64": unet.seresnext50_unet64,
        "hrnet18_unet64": unet.hrnet18_unet64,
        "hrnet34_unet64": unet.hrnet34_unet64,
        "hrnet48_unet64": unet.hrnet48_unet64,
        "densenet121_unet64": unet.densenet121_unet64,
        "densenet121_unet128": unet.densenet121_unet128,
        "densenet121_unet64_offset": densenet121_unet64_offset,
        "sexresnet50_unet128": sexresnet50_unet128,
        "xresnet50_unet128": xresnet50_unet128,
        # Residual UNnet
        "seresnext50_runet64": runet.seresnext50_runet64,
        "hrnet18_runet64": runet.hrnet18_runet64,
        "hrnet34_runet64": runet.hrnet34_runet64,
        "hrnet48_runet64": runet.hrnet48_runet64,
        "densenet121_runet64": runet.densenet121_runet64,
        # Deeplab
        "resnet34_deeplab128": deeplab.resnet34_deeplab128,
        "seresnext101_deeplab128": deeplab.seresnext101_deeplab256,
        # HRNet
        "hrnet18": hrnet.hrnet18,
        "hrnet34": hrnet.hrnet34,
        "hrnet48": hrnet.hrnet48,
        "hg8": hg.hg8,
        "hg4": hg.hg4,
        "shg4": hg.shg4,
    }

    return registry[model_name.lower()](dropout=dropout, pretrained=pretrained)
