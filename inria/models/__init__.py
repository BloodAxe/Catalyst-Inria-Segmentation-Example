from torch import nn

from . import fpn, unet, deeplab, hrnet, hg

__all__ = ["get_model", "MODEL_REGISTRY"]

MODEL_REGISTRY = {
    # FPN family using concatenation
    "resnet34_fpncat128": fpn.resnet34_fpncat128,
    "resnet152_fpncat256": fpn.resnet152_fpncat256,
    "seresnext50_fpncat128": fpn.seresnext50_fpncat128,
    "seresnext101_fpncat256": fpn.seresnext101_fpncat256,
    # FPN family using summation
    "seresnext101_fpnsum256": fpn.seresnext101_fpnsum256,
    # UNet
    "resnet18_unet32": unet.resnet18_unet32,
    "resnet34_unet32": unet.resnet34_unet32,
    "resnet50_unet32": unet.resnet50_unet32,
    "resnet101_unet32": unet.resnet101_unet32,
    "resnet152_unet32": unet.resnet152_unet32,
    # Deeplab
    "resnet34_deeplab128": deeplab.resnet34_deeplab128,
    "seresnext101_deeplab128": deeplab.seresnext101_deeplab256,
    # HRNet
    "hrnet18": hrnet.hrnet18,
    "hrnet34": hrnet.hrnet34,
    "hrnet48": hrnet.hrnet48,
    # Hourglass
    "hg8": hg.hg8,
    "hg4": hg.hg4,
    "shg4": hg.shg4,
    "shg8": hg.shg8,
}


def get_model(model_name: str, dropout=0.0, pretrained=True) -> nn.Module:
    return MODEL_REGISTRY[model_name.lower()](dropout=dropout, pretrained=pretrained)
