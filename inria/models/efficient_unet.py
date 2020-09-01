from collections import OrderedDict
from functools import partial
from typing import Union, List, Dict

from pytorch_toolbelt.modules import conv1x1, UnetBlock, ACT_RELU, ABN, ACT_SWISH, Swish, DropBlock2D
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
    def __init__(self, in_channels: int, out_channels: int, activation=Swish, drop_path_rate=0.0):
        super().__init__()
        self.ir = InvertedResidual(in_channels, out_channels, act_layer=activation, se_ratio=0.25, exp_ratio=4)
        self.drop = DropBlock2D(drop_path_rate, 2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation(inplace=True),
        )

    def forward(self, x):
        x = self.ir(x)
        x = self.drop(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class EfficientUNetDecoder(UNetDecoder):
    def __init__(
        self,
        feature_maps: List[int],
        decoder_features: List[int],
        upsample_block=nn.UpsamplingNearest2d,
        activation=Swish,
    ):
        super().__init__(
            feature_maps,
            unet_block=partial(EfficientUnetBlock, activation=activation, drop_path_rate=0.2),
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
        activation=Swish,
    ):
        super().__init__()
        self.encoder = encoder

        self.decoder = EfficientUNetDecoder(
            feature_maps=encoder.channels, decoder_features=unet_channels, activation=activation
        )

        self.mask = nn.Sequential(
            OrderedDict([("drop", nn.Dropout2d(dropout)), ("conv", conv1x1(self.decoder.channels[0], num_classes))])
        )

        self.full_size_mask = full_size_mask

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x_size = x.size()
        enc = self.encoder(x)
        dec = self.decoder(enc)

        # Decode mask
        mask = self.mask(dec[0])

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
        encoder, num_classes=num_classes, unet_channels=[32, 64, 128, 256], activation=Swish, dropout=dropout
    )
