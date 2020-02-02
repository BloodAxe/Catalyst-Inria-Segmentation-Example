from functools import partial
from typing import Union, Callable, List

import torch
from pytorch_toolbelt.modules import ABN, ACT_RELU
from pytorch_toolbelt.modules import encoders as E
from pytorch_toolbelt.modules.decoders import DecoderModule
from pytorch_toolbelt.modules.encoders import EncoderModule
from torch import nn
from torch.nn import functional as F

from ..dataset import OUTPUT_MASK_KEY

__all__ = ["seresnext50_unet64", "hrnet18_unet64", "hrnet34_unet64", "hrnet48_unet64", "densenet121_unet64"]


class ConvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True))

    def forward(self, dec, enc):
        x = torch.cat([dec, enc], dim=1)
        return self.seq(x)


class UnetDecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2), nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


class UNetDecoderV2(DecoderModule):
    def __init__(
        self,
        feature_maps: List[int],
        decoder_features: List[int],
        mask_channels: int,
        last_upsample_filters=None,
        dropout=0.0,
        abn_block=ABN,
    ):
        super().__init__()

        if not isinstance(decoder_features, list):
            decoder_features = [decoder_features * (2 ** i) for i in range(len(feature_maps))]

        if last_upsample_filters is None:
            last_upsample_filters = decoder_features[0]

        self.encoder_features = feature_maps
        self.decoder_features = decoder_features
        self.decoder_stages = nn.ModuleList([self.get_decoder(idx) for idx in range(0, len(self.decoder_features))])

        self.bottlenecks = nn.ModuleList(
            [
                ConvBottleneck(self.encoder_features[-i - 2] + f, f)
                for i, f in enumerate(reversed(self.decoder_features[:]))
            ]
        )

        self.output_filters = decoder_features

        self.last_upsample = UnetDecoderBlock(decoder_features[0], last_upsample_filters, last_upsample_filters)

        self.final = nn.Conv2d(last_upsample_filters, mask_channels, kernel_size=1)

    def get_decoder(self, layer):
        in_channels = (
            self.encoder_features[layer + 1]
            if layer + 1 == len(self.decoder_features)
            else self.decoder_features[layer + 1]
        )
        return UnetDecoderBlock(in_channels, self.decoder_features[layer], self.decoder_features[max(layer, 0)])

    def forward(self, feature_maps):

        last_dec_out = feature_maps[-1]

        x = last_dec_out
        for idx, bottleneck in enumerate(self.bottlenecks):
            rev_idx = -(idx + 1)
            decoder = self.decoder_stages[rev_idx]
            x = decoder(x)
            x = bottleneck(x, feature_maps[rev_idx - 1])

        x = self.last_upsample(x)

        f = self.final(x)

        return f


class UnetV2SegmentationModel(nn.Module):
    def __init__(
        self,
        encoder: EncoderModule,
        num_classes: int,
        unet_channels: Union[int, List[int]],
        last_upsample_filters=None,
        dropout=0.25,
        abn_block: Union[ABN, Callable[[int], nn.Module]] = ABN,
        full_size_mask=True,
    ):
        super().__init__()
        self.encoder = encoder

        self.decoder = UNetDecoderV2(
            feature_maps=encoder.output_filters,
            decoder_features=unet_channels,
            last_upsample_filters=last_upsample_filters,
            mask_channels=num_classes,
            dropout=dropout,
            abn_block=abn_block,
        )

        self.full_size_mask = full_size_mask

    def forward(self, x):
        features = self.encoder(x)

        # Decode mask
        mask = self.decoder(features)

        if self.full_size_mask:
            mask = F.interpolate(mask, size=x.size()[2:], mode="bilinear", align_corners=False)

        output = {OUTPUT_MASK_KEY: mask}
        return output


def seresnext50_unet64(input_channels=3, num_classes=1, dropout=0.0, pretrained=True):
    encoder = E.SEResNeXt50Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetV2SegmentationModel(
        encoder,
        num_classes=num_classes,
        unet_channels=[64, 128, 256, 256],
        dropout=dropout,
        abn_block=partial(ABN, activation=ACT_RELU),
    )


def hrnet18_unet64(input_channels=3, num_classes=1, dropout=0.0, pretrained=True):
    encoder = E.HRNetV2Encoder18(pretrained=pretrained, layers=[1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetV2SegmentationModel(
        encoder,
        num_classes=num_classes,
        unet_channels=[64, 128, 256],
        dropout=dropout,
        abn_block=partial(ABN, activation=ACT_RELU),
    )


def hrnet34_unet64(input_channels=3, num_classes=1, dropout=0.0, pretrained=True):
    encoder = E.HRNetV2Encoder34(pretrained=pretrained, layers=[1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetV2SegmentationModel(
        encoder,
        num_classes=num_classes,
        unet_channels=[128, 128, 256],
        last_upsample_filters=64,
        dropout=dropout,
        abn_block=partial(ABN, activation=ACT_RELU),
    )


def hrnet48_unet64(input_channels=3, num_classes=1, dropout=0.0, pretrained=True):
    encoder = E.HRNetV2Encoder48(pretrained=pretrained, layers=[1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetV2SegmentationModel(
        encoder,
        num_classes=num_classes,
        unet_channels=[128, 128, 256],
        last_upsample_filters=64,
        dropout=dropout,
        abn_block=partial(ABN, activation=ACT_RELU),
    )


def densenet121_unet64(input_channels=3, num_classes=1, dropout=0.0, pretrained=True):
    encoder = E.DenseNet121Encoder(pretrained=pretrained, layers=[1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetV2SegmentationModel(
        encoder,
        num_classes=num_classes,
        unet_channels=[128, 128, 256],
        last_upsample_filters=64,
        dropout=dropout,
        abn_block=partial(ABN, activation=ACT_RELU),
    )


def densenet121_unet128(input_channels=3, num_classes=1, dropout=0.0, pretrained=True):
    encoder = E.DenseNet121Encoder(pretrained=pretrained, layers=[1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetV2SegmentationModel(
        encoder,
        num_classes=num_classes,
        unet_channels=[128, 128, 256],
        last_upsample_filters=128,
        dropout=dropout,
        abn_block=partial(ABN, activation=ACT_RELU),
    )
