from functools import partial

from torch import nn

from inria.models.fpn import effnetB4_fpn, reset101_fpn, reset34_fpn, effnetB7_fpn
from inria.models.linknet import LinkNet34, LinkNet152
from inria.models.unet import UNet

from pytorch_toolbelt.modules import encoders as E

__all__ = ['get_model']


def get_model(model_name: str) -> nn.Module:
    registry = {
        # Unet family
        'unet': partial(UNet, upsample=False),

        # Linknet family
        'linknet34': LinkNet34,
        'linknet152': LinkNet152,

        # FPN family
        'reset34_fpn': reset34_fpn,
        'reset101_fpn': reset101_fpn,
        'effnetb4_fpn': effnetB4_fpn,
        'effnetb7_fpn': effnetB7_fpn,

    }

    return registry[model_name.lower()]()
