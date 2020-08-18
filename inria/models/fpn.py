from collections import OrderedDict

from pytorch_toolbelt.modules import ABN, conv1x1
from pytorch_toolbelt.modules import encoders as E
from pytorch_toolbelt.modules.decoders import FPNSumDecoder, FPNCatDecoder
from pytorch_toolbelt.modules.encoders import EncoderModule
from torch import nn
from torch.nn import functional as F

from ..dataset import OUTPUT_MASK_KEY
from catalyst.registry import Model

__all__ = [
    "FPNSumSegmentationModel",
    "FPNCatSegmentationModel",
    "resnet34_fpncat128",
    "resnet152_fpncat256",
    "seresnext50_fpncat128",
    "seresnext101_fpncat256",
    "seresnext101_fpnsum256",
    "effnetB4_fpncat128",
]


class FPNSumSegmentationModel(nn.Module):
    def __init__(
        self, encoder: EncoderModule, num_classes: int, dropout=0.25, full_size_mask=True, fpn_channels=256,
    ):
        super().__init__()
        self.encoder = encoder

        self.decoder = FPNSumDecoder(feature_maps=encoder.output_filters, fpn_channels=fpn_channels,)
        self.mask = nn.Sequential(
            OrderedDict([("drop", nn.Dropout2d(dropout)), ("conv", conv1x1(fpn_channels, num_classes))])
        )

        self.full_size_mask = full_size_mask

    def forward(self, x):
        x_size = x.size()
        x = self.encoder(x)
        x = self.decoder(x)

        # Decode mask
        mask = self.mask(x[0])

        if self.full_size_mask:
            mask = F.interpolate(mask, size=x_size[2:], mode="bilinear", align_corners=False)

        output = {
            OUTPUT_MASK_KEY: mask,
        }

        return output


class FPNCatSegmentationModel(nn.Module):
    def __init__(
        self, encoder: EncoderModule, num_classes: int, dropout=0.25, fpn_channels=256, full_size_mask=True,
    ):
        super().__init__()
        self.encoder = encoder

        self.decoder = FPNCatDecoder(feature_maps=encoder.output_filters, fpn_channels=fpn_channels)
        self.mask = nn.Sequential(
            OrderedDict([("drop", nn.Dropout2d(dropout)), ("conv", conv1x1(fpn_channels, num_classes))])
        )
        self.full_size_mask = full_size_mask

    def forward(self, x):
        x_size = x.size()
        x = self.encoder(x)
        x = self.decoder(x)

        # Decode mask
        mask = self.mask(x[0])

        if self.full_size_mask:
            mask = F.interpolate(mask, size=x_size[2:], mode="bilinear", align_corners=False)

        output = {
            OUTPUT_MASK_KEY: mask,
        }

        return output


@Model
def resnet34_fpncat128(num_classes=5, dropout=0.0, pretrained=True):
    encoder = E.Resnet34Encoder(pretrained=pretrained)
    return FPNCatSegmentationModel(encoder, num_classes=num_classes, fpn_channels=128, dropout=dropout)


@Model
def seresnext50_fpncat128(num_classes=5, dropout=0.0, pretrained=True):
    encoder = E.SEResNeXt50Encoder(pretrained=pretrained)
    return FPNCatSegmentationModel(encoder, num_classes=num_classes, fpn_channels=128, dropout=dropout)


@Model
def seresnext101_fpncat256(num_classes=5, dropout=0.0, pretrained=True):
    encoder = E.SEResNeXt101Encoder(pretrained=pretrained)
    return FPNCatSegmentationModel(encoder, num_classes=num_classes, fpn_channels=256, dropout=dropout)


@Model
def seresnext101_fpnsum256(num_classes=5, dropout=0.0, pretrained=True):
    encoder = E.SEResNeXt101Encoder(pretrained=pretrained)
    return FPNSumSegmentationModel(encoder, num_classes=num_classes, fpn_channels=256, dropout=dropout)


@Model
def resnet152_fpncat256(num_classes=5, dropout=0.0, pretrained=True):
    encoder = E.Resnet152Encoder(pretrained=pretrained)
    return FPNCatSegmentationModel(encoder, num_classes=num_classes, fpn_channels=256, dropout=dropout)


@Model
def effnetB4_fpncat128(num_classes=5, dropout=0.0, pretrained=True):
    encoder = E.EfficientNetB4Encoder(abn_params={"activation": "swish"}, pretrained=pretrained)
    return FPNCatSegmentationModel(encoder, num_classes=num_classes, fpn_channels=128, dropout=dropout)
