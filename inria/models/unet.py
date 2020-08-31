from collections import OrderedDict
from functools import partial
from typing import Union, List, Dict

from pytorch_toolbelt.modules import conv1x1, UnetBlock, ACT_RELU, ABN, ACT_SWISH
from pytorch_toolbelt.modules import encoders as E
from pytorch_toolbelt.modules.decoders import UNetDecoder
from pytorch_toolbelt.modules.encoders import EncoderModule
from .timm_encoders import B4Encoder, B0Encoder, B6Encoder
from torch import nn, Tensor
from torch.nn import functional as F

from ..dataset import OUTPUT_MASK_KEY, output_mask_name_for_stride
from catalyst.registry import Model

__all__ = [
    "UnetSegmentationModel",
    "resnet18_unet32",
    "resnet34_unet32",
    "resnet50_unet32",
    "resnet101_unet64",
    "resnet152_unet32",
    "densenet121_unet32",
    "densenet161_unet32",
    "densenet169_unet32",
    "densenet201_unet32",
    "b0_unet32_s2",
    "b4_unet32",
    "b6_unet32_s2",
    "b6_unet32_s2_bi",
    "b6_unet32_s2_tc",
    "b6_unet32_s2_rdtc",
]


class UnetSegmentationModel(nn.Module):
    def __init__(
        self,
        encoder: EncoderModule,
        unet_channels: Union[int, List[int]],
        num_classes: int = 1,
        dropout=0.25,
        full_size_mask=True,
        activation=ACT_RELU,
        upsample_block=nn.UpsamplingNearest2d,
        need_supervision_masks=False,
    ):
        super().__init__()
        self.encoder = encoder

        abn_block = partial(ABN, activation=activation)
        self.decoder = UNetDecoder(
            feature_maps=encoder.channels,
            decoder_features=unet_channels,
            unet_block=partial(UnetBlock, abn_block=abn_block),
            upsample_block=upsample_block,
        )

        self.mask = nn.Sequential(
            OrderedDict([("drop", nn.Dropout2d(dropout)), ("conv", conv1x1(unet_channels[0], num_classes))])
        )

        if need_supervision_masks:
            self.supervision = nn.ModuleList([conv1x1(channels, num_classes) for channels in self.decoder.channels])
            self.supervision_names = [output_mask_name_for_stride(stride) for stride in self.encoder.strides]
        else:
            self.supervision = None
            self.supervision_names = None

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

        if self.supervision is not None:
            for feature_map, supervision, name in zip(x, self.supervision, self.supervision_names):
                output[name] = supervision(feature_map)

        return output


@Model
def resnet18_unet32(input_channels=3, num_classes=1, dropout=0.0, pretrained=True):
    encoder = E.Resnet18Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetSegmentationModel(encoder, num_classes=num_classes, unet_channels=[32, 64, 128, 256], dropout=dropout)


@Model
def resnet34_unet32(input_channels=3, num_classes=1, dropout=0.0, pretrained=True):
    encoder = E.Resnet34Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetSegmentationModel(encoder, num_classes=num_classes, unet_channels=[32, 64, 128, 256], dropout=dropout)


@Model
def resnet50_unet32(input_channels=3, num_classes=1, dropout=0.0, pretrained=True):
    encoder = E.Resnet50Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetSegmentationModel(encoder, num_classes=num_classes, unet_channels=[32, 64, 128, 256], dropout=dropout)


@Model
def resnet101_unet64(input_channels=3, num_classes=1, dropout=0.5, pretrained=True):
    encoder = E.Resnet101Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetSegmentationModel(encoder, num_classes=num_classes, unet_channels=[64, 128, 256, 512], dropout=dropout)


@Model
def resnet152_unet32(input_channels=3, num_classes=1, dropout=0.0, pretrained=True):
    encoder = E.Resnet152Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetSegmentationModel(encoder, num_classes=num_classes, unet_channels=[32, 64, 128, 256], dropout=dropout)


# Densenets


@Model
def densenet121_unet32(input_channels=3, num_classes=1, dropout=0.0, pretrained=True):
    encoder = E.DenseNet121Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetSegmentationModel(encoder, num_classes=num_classes, unet_channels=[32, 64, 128, 256], dropout=dropout)


@Model
def densenet161_unet32(input_channels=3, num_classes=1, dropout=0.0, pretrained=True):
    encoder = E.DenseNet161Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetSegmentationModel(encoder, num_classes=num_classes, unet_channels=[32, 64, 128, 256], dropout=dropout)


@Model
def densenet169_unet32(input_channels=3, num_classes=1, dropout=0.0, pretrained=True):
    encoder = E.DenseNet169Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetSegmentationModel(encoder, num_classes=num_classes, unet_channels=[32, 64, 128, 256], dropout=dropout)


@Model
def densenet201_unet32(input_channels=3, num_classes=1, dropout=0.0, pretrained=True):
    encoder = E.DenseNet201Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetSegmentationModel(encoder, num_classes=num_classes, unet_channels=[32, 64, 128, 256], dropout=dropout)


# HRNet


@Model
def hrnet18_unet32(input_channels=3, num_classes=1, dropout=0.0, pretrained=True):
    encoder = E.HRNetV2Encoder18(pretrained=pretrained)
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetSegmentationModel(encoder, num_classes=num_classes, unet_channels=[32, 64, 128, 256], dropout=dropout)


@Model
def hrnet34_unet32(input_channels=3, num_classes=1, dropout=0.0, pretrained=True):
    encoder = E.HRNetV2Encoder34(pretrained=pretrained)
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetSegmentationModel(encoder, num_classes=num_classes, unet_channels=[32, 64, 128, 256], dropout=dropout)


@Model
def hrnet48_unet32(input_channels=3, num_classes=1, dropout=0.0, pretrained=True):
    encoder = E.HRNetV2Encoder48(pretrained=pretrained)
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetSegmentationModel(encoder, num_classes=num_classes, unet_channels=[32, 64, 128, 256], dropout=dropout)


# B0-Unet
@Model
def b0_unet32_s2(input_channels=3, num_classes=1, dropout=0.1, pretrained=True):
    encoder = B0Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetSegmentationModel(
        encoder, num_classes=num_classes, unet_channels=[16, 32, 64, 128], activation=ACT_SWISH, dropout=dropout
    )


@Model
def b4_unet32(input_channels=3, num_classes=1, dropout=0.2, pretrained=True):
    encoder = B4Encoder(pretrained=pretrained)
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetSegmentationModel(
        encoder, num_classes=num_classes, unet_channels=[32, 64, 128], activation=ACT_SWISH, dropout=dropout
    )


@Model
def b4_unet32_s2(input_channels=3, num_classes=1, dropout=0.2, pretrained=True):
    encoder = B4Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetSegmentationModel(
        encoder, num_classes=num_classes, unet_channels=[32, 64, 128, 256], activation=ACT_SWISH, dropout=dropout
    )


@Model
def b6_unet32_s2(input_channels=3, num_classes=1, dropout=0.2, pretrained=True):
    encoder = B6Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetSegmentationModel(
        encoder, num_classes=num_classes, unet_channels=[32, 64, 128, 256], activation=ACT_SWISH, dropout=dropout
    )


@Model
def b6_unet32_s2_bi(input_channels=3, num_classes=1, dropout=0.2, pretrained=True):
    encoder = B6Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetSegmentationModel(
        encoder,
        num_classes=num_classes,
        unet_channels=[32, 64, 128, 256],
        activation=ACT_SWISH,
        dropout=dropout,
        upsample_block=nn.UpsamplingBilinear2d,
    )


@Model
def b6_unet32_s2_tc(input_channels=3, num_classes=1, dropout=0.2, pretrained=True):
    encoder = B6Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    from pytorch_toolbelt.modules.upsample import DeconvolutionUpsample2d

    return UnetSegmentationModel(
        encoder,
        num_classes=num_classes,
        unet_channels=[32, 64, 128, 256],
        activation=ACT_SWISH,
        dropout=dropout,
        upsample_block=DeconvolutionUpsample2d,
    )


@Model
def b6_unet32_s2_rdtc(input_channels=3, num_classes=1, dropout=0.2, need_supervision_masks=False, pretrained=True):
    encoder = B6Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    from pytorch_toolbelt.modules.upsample import ResidualDeconvolutionUpsample2d

    return UnetSegmentationModel(
        encoder,
        num_classes=num_classes,
        unet_channels=[32, 64, 128, 256],
        activation=ACT_SWISH,
        dropout=dropout,
        need_supervision_masks=need_supervision_masks,
        upsample_block=ResidualDeconvolutionUpsample2d,
    )
