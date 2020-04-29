from collections import OrderedDict
from typing import Union, List, Dict

from pytorch_toolbelt.modules import conv1x1
from pytorch_toolbelt.modules import encoders as E
from pytorch_toolbelt.modules.decoders import UNetDecoder
from pytorch_toolbelt.modules.encoders import EncoderModule
from torch import nn, Tensor
from torch.nn import functional as F

from ..dataset import OUTPUT_MASK_KEY

__all__ = [
    "UnetSegmentationModel",
    "resnet18_unet32",
    "resnet34_unet32",
    "resnet50_unet32",
    "resnet101_unet64",
    "resnet152_unet64",
    "densenet121_unet32",
    "densenet161_unet32",
    "densenet169_unet32",
    "densenet201_unet32"
]


class UnetSegmentationModel(nn.Module):
    def __init__(
            self,
            encoder: EncoderModule,
            unet_channels: Union[int, List[int]],
            num_classes: int = 1,
            dropout=0.25,
            full_size_mask=True,
    ):
        super().__init__()
        self.encoder = encoder

        self.decoder = UNetDecoder(
            feature_maps=encoder.channels,
            decoder_features=unet_channels,
            upsample_block=nn.UpsamplingNearest2d
        )

        self.mask = nn.Sequential(OrderedDict([
            ("drop", nn.Dropout2d(dropout)),
            ("conv", conv1x1(unet_channels[0], num_classes))]))

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


def resnet18_unet32(input_channels=3, num_classes=1, dropout=0.0, pretrained=True):
    encoder = E.Resnet18Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetSegmentationModel(
        encoder,
        num_classes=num_classes,
        unet_channels=[32, 64, 128, 256],
        dropout=dropout,
    )


def resnet34_unet32(input_channels=3, num_classes=1, dropout=0.0, pretrained=True):
    encoder = E.Resnet34Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetSegmentationModel(
        encoder,
        num_classes=num_classes,
        unet_channels=[32, 64, 128, 256],
        dropout=dropout,
    )


def resnet50_unet32(input_channels=3, num_classes=1, dropout=0.0, pretrained=True):
    encoder = E.Resnet50Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetSegmentationModel(
        encoder,
        num_classes=num_classes,
        unet_channels=[32, 64, 128, 256],
        dropout=dropout,
    )


def resnet101_unet64(input_channels=3, num_classes=1, dropout=0.0, pretrained=True):
    encoder = E.Resnet50Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetSegmentationModel(
        encoder,
        num_classes=num_classes,
        unet_channels=[32, 64, 128, 256],
        dropout=dropout,
    )


def resnet152_unet64(input_channels=3, num_classes=1, dropout=0.0, pretrained=True):
    encoder = E.Resnet152Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetSegmentationModel(
        encoder,
        num_classes=num_classes,
        unet_channels=[32, 64, 128, 256],
        dropout=dropout,
    )


# Densenets

def densenet121_unet32(input_channels=3, num_classes=1, dropout=0.0, pretrained=True):
    encoder = E.DenseNet121Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetSegmentationModel(
        encoder,
        num_classes=num_classes,
        unet_channels=[32, 64, 128, 256],
        dropout=dropout,
    )


def densenet161_unet32(input_channels=3, num_classes=1, dropout=0.0, pretrained=True):
    encoder = E.DenseNet161Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetSegmentationModel(
        encoder,
        num_classes=num_classes,
        unet_channels=[32, 64, 128, 256],
        dropout=dropout,
    )


def densenet169_unet32(input_channels=3, num_classes=1, dropout=0.0, pretrained=True):
    encoder = E.DenseNet169Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetSegmentationModel(
        encoder,
        num_classes=num_classes,
        unet_channels=[32, 64, 128, 256],
        dropout=dropout,
    )


def densenet201_unet32(input_channels=3, num_classes=1, dropout=0.0, pretrained=True):
    encoder = E.DenseNet201Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetSegmentationModel(
        encoder,
        num_classes=num_classes,
        unet_channels=[32, 64, 128, 256],
        dropout=dropout,
    )


# HRNet

def hrnet18_unet32(input_channels=3, num_classes=1, dropout=0.0, pretrained=True):
    encoder = E.HRNetV2Encoder18(pretrained=pretrained)
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetSegmentationModel(
        encoder,
        num_classes=num_classes,
        unet_channels=[32, 64, 128, 256],
        dropout=dropout,
    )


def hrnet34_unet32(input_channels=3, num_classes=1, dropout=0.0, pretrained=True):
    encoder = E.HRNetV2Encoder34(pretrained=pretrained)
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetSegmentationModel(
        encoder,
        num_classes=num_classes,
        unet_channels=[32, 64, 128, 256],
        dropout=dropout,
    )


def hrnet48_unet32(input_channels=3, num_classes=1, dropout=0.0, pretrained=True):
    encoder = E.HRNetV2Encoder48(pretrained=pretrained)
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetSegmentationModel(
        encoder,
        num_classes=num_classes,
        unet_channels=[32, 64, 128, 256],
        dropout=dropout,
    )
