from torch import nn

from . import fpn, unet, deeplab, hrnet, hg, can

__all__ = ["get_model"]


def get_model(model_name: str, dropout=0.0, pretrained=True) -> nn.Module:
    from catalyst.dl import registry

    model_fn = registry.MODEL.get(model_name)
    return model_fn(dropout=dropout, pretrained=pretrained)
