from typing import Tuple, Dict

import torch
from torch import nn

from . import fpn, unet, deeplab, hrnet, hg, can, efficient_unet, u2net

__all__ = ["get_model", "model_from_checkpoint"]


def get_model(model_name: str, pretrained=True, **kwargs) -> nn.Module:
    from catalyst.dl import registry

    model_fn = registry.MODEL.get(model_name)
    return model_fn(pretrained=pretrained, **kwargs)


def model_from_checkpoint(checkpoint_name: str, strict=True, **kwargs) -> Tuple[nn.Module, Dict]:
    checkpoint = torch.load(checkpoint_name, map_location="cpu")
    model_state_dict = checkpoint["model_state_dict"]
    model_name = checkpoint["checkpoint_data"]["cmd_args"]["model"]

    model = get_model(model_name, pretrained=False, **kwargs)
    model.load_state_dict(model_state_dict, strict=strict)

    return model.eval(), checkpoint
