from collections import OrderedDict
from functools import partial
from typing import Union, List, Dict

from pytorch_toolbelt.modules import conv1x1, UnetBlock, ACT_RELU, ABN, ACT_SWISH, Swish
from pytorch_toolbelt.modules import encoders as E
from pytorch_toolbelt.modules.decoders import UNetDecoder
from pytorch_toolbelt.modules.encoders import EncoderModule
from .timm_encoders import B4Encoder, B0Encoder, B6Encoder
from torch import nn, Tensor
from torch.nn import functional as F
from timm.models.efficientnet_blocks import InvertedResidual
from ..dataset import OUTPUT_MASK_KEY
from catalyst.registry import Model

__all__ = [
    "EfficientUnetBlock",
    "EfficientUnetSegmentationModel",
    "b4_effunet32_s2",
]


class EfficientUnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, abn_block=ABN):
        super().__init__()
        mid_channels = (in_channels + out_channels) // 2
        self.conv1 = InvertedResidual(in_channels, mid_channels, act_layer=Swish, se_ratio=0.25, exp_ratio=4)
        self.conv2 = InvertedResidual(mid_channels, out_channels, act_layer=Swish, se_ratio=0.25, exp_ratio=4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class EfficientUNetDecoder(UNetDecoder):
    def __init__(self, feature_maps: List[int], decoder_features: List[int], upsample_block=nn.UpsamplingNearest2d):
        super().__init__(
            feature_maps,
            unet_block=EfficientUnetBlock,
            decoder_features=decoder_features,
            upsample_block=upsample_block,
        )


class EfficientUnetSegmentationModel(nn.Module):
    def __init__(
        self,
        encoder: EncoderModule,
        unet_channels: Union[int, List[int]],
        num_classes: int = 1,
        dropout=0.25,
        full_size_mask=True,
        activation=ACT_RELU,
    ):
        super().__init__()
        self.encoder = encoder

        self.decoder = EfficientUNetDecoder(feature_maps=encoder.channels, decoder_features=unet_channels)

        self.mask = nn.Sequential(
            OrderedDict([("drop", nn.Dropout2d(dropout)), ("conv", conv1x1(self.decoder.channels[0], num_classes))])
        )

        self.full_size_mask = full_size_mask

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x_size = x.size()
        x = self.encoder(x)
        x = self.decoder(x)

        # Decode mask
        mask = self.mask(x[0])

        if self.full_size_mask:
            mask = F.interpolate(mask, size=x_size[2:], mode="bilinear", align_corners=False)

        output = {OUTPUT_MASK_KEY: mask}
        return output


@Model
def b4_effunet32_s2(input_channels=3, num_classes=1, dropout=0.2, pretrained=True):
    encoder = B4Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return EfficientUnetSegmentationModel(
        encoder, num_classes=num_classes, unet_channels=[32, 64, 128, 256], activation=ACT_SWISH, dropout=dropout
    )
