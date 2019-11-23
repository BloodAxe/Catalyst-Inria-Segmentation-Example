import torch
from pytorch_toolbelt.inference.functional import pad_image_tensor, unpad_image_tensor
from pytorch_toolbelt.modules import ABN
from pytorch_toolbelt.modules import encoders as E
from pytorch_toolbelt.modules.decoders import FPNSumDecoder, FPNCatDecoder
from pytorch_toolbelt.modules.encoders import EncoderModule
from torch import nn
from torch.nn import functional as F

from ..dataset import OUTPUT_MASK_4_KEY, OUTPUT_MASK_8_KEY, OUTPUT_MASK_16_KEY, OUTPUT_MASK_32_KEY, OUTPUT_MASK_KEY

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
        self,
        encoder: EncoderModule,
        num_classes: int,
        dropout=0.25,
        abn_block=ABN,
        full_size_mask=True,
        fpn_channels=256,
    ):
        super().__init__()
        self.encoder = encoder

        self.decoder = FPNSumDecoder(
            feature_maps=encoder.output_filters,
            num_classes=num_classes,
            fpn_channels=fpn_channels,
            abn_block=abn_block,
            dropout=dropout,
        )

        self.full_size_mask = full_size_mask

    def forward(self, x):
        x, pad = pad_image_tensor(x, 32)

        enc_features = self.encoder(x)

        # Decode mask
        mask, dsv = self.decoder(enc_features)

        if self.full_size_mask:
            mask = F.interpolate(mask, size=x.size()[2:], mode="bilinear", align_corners=False)
            mask = unpad_image_tensor(mask, pad)

        output = {
            OUTPUT_MASK_KEY: mask,
            OUTPUT_MASK_4_KEY: dsv[3],
            OUTPUT_MASK_8_KEY: dsv[2],
            OUTPUT_MASK_16_KEY: dsv[1],
            OUTPUT_MASK_32_KEY: dsv[0],
        }

        return output


class FPNCatSegmentationModel(nn.Module):
    def __init__(
        self,
        encoder: EncoderModule,
        num_classes: int,
        dropout=0.25,
        abn_block=ABN,
        fpn_channels=256,
        full_size_mask=True,
    ):
        super().__init__()
        self.encoder = encoder

        self.decoder = FPNCatDecoder(
            feature_maps=encoder.output_filters,
            num_classes=num_classes,
            fpn_channels=fpn_channels,
            abn_block=abn_block,
            dropout=dropout,
        )

        self.full_size_mask = full_size_mask

    def forward(self, x):
        x, pad = pad_image_tensor(x, 32)
        enc_features = self.encoder(x)

        # Decode mask
        mask, dsv = self.decoder(enc_features)

        if self.full_size_mask:
            mask = F.interpolate(mask, size=x.size()[2:], mode="bilinear", align_corners=False)
            mask = unpad_image_tensor(mask, pad)

        output = {
            OUTPUT_MASK_KEY: mask,
            OUTPUT_MASK_4_KEY: dsv[0],
            OUTPUT_MASK_8_KEY: dsv[1],
            OUTPUT_MASK_16_KEY: dsv[2],
            OUTPUT_MASK_32_KEY: dsv[3],
        }

        return output


class FPNCatSegmentationModel(nn.Module):
    def __init__(
        self,
        encoder: EncoderModule,
        num_classes: int,
        dropout=0.25,
        abn_block=ABN,
        fpn_channels=256,
        full_size_mask=True,
    ):
        super().__init__()
        self.encoder = encoder

        self.decoder = FPNCatDecoder(
            feature_maps=encoder.output_filters,
            num_classes=num_classes,
            fpn_channels=fpn_channels,
            abn_block=abn_block,
            dropout=dropout,
        )

        self.full_size_mask = full_size_mask

    def forward(self, x):
        x, pad = pad_image_tensor(x, 32)
        enc_features = self.encoder(x)

        # Decode mask
        mask, dsv = self.decoder(enc_features)

        if self.full_size_mask:
            mask = F.interpolate(mask, size=x.size()[2:], mode="bilinear", align_corners=False)
            mask = unpad_image_tensor(mask, pad)

        output = {
            OUTPUT_MASK_KEY: mask,
            OUTPUT_MASK_4_KEY: dsv[0],
            OUTPUT_MASK_8_KEY: dsv[1],
            OUTPUT_MASK_16_KEY: dsv[2],
            OUTPUT_MASK_32_KEY: dsv[3],
        }

        return output


class RFPNCatSegmentationModel(nn.Module):
    """
    Segmentation model with recurrent FPN decoder
    """

    def __init__(
        self,
        encoder: EncoderModule,
        num_classes: int,
        dropout=0.25,
        abn_block=ABN,
        fpn_channels=256,
        full_size_mask=True,
    ):
        super().__init__()
        self.encoder = encoder

        self.decoder = FPNCatDecoder(
            feature_maps=[f + num_classes for f in encoder.output_filters],
            num_classes=num_classes,
            fpn_channels=fpn_channels,
            abn_block=abn_block,
            dropout=dropout,
        )
        self.num_classes = num_classes
        self.full_size_mask = full_size_mask

    def forward(self, input):
        input, pad = pad_image_tensor(input, 32)
        enc_features = self.encoder(input)

        mask = torch.zeros(
            (enc_features[0].size(0), self.num_classes, enc_features[0].size(2), enc_features[0].size(3)),
            device=enc_features[0].device,
        )

        for i in range(3):
            # Concatenate intermediate mask with encoder feature maps
            x = [
                torch.cat([f, F.interpolate(mask, size=f.size()[2:], mode="bilinear", align_corners=False)], dim=1)
                for f in enc_features
            ]
            mask, dsv = self.decoder(x)

        if self.full_size_mask:
            mask = F.interpolate(mask, size=input.size()[2:], mode="bilinear", align_corners=False)
            mask = unpad_image_tensor(mask, pad)

        output = {
            OUTPUT_MASK_KEY: mask,
            OUTPUT_MASK_4_KEY: dsv[0],
            OUTPUT_MASK_8_KEY: dsv[1],
            OUTPUT_MASK_16_KEY: dsv[2],
            OUTPUT_MASK_32_KEY: dsv[3],
        }

        return output


def resnet34_fpncat128(num_classes=1, dropout=0.0):
    encoder = E.Resnet34Encoder()
    return FPNCatSegmentationModel(encoder, num_classes=num_classes, fpn_channels=128, dropout=dropout)


def resnet34_rfpncat128(num_classes=1, dropout=0.0):
    encoder = E.Resnet34Encoder()
    return RFPNCatSegmentationModel(encoder, num_classes=num_classes, fpn_channels=128, dropout=dropout)


def seresnext50_fpncat128(num_classes=1, dropout=0.0):
    encoder = E.SEResNeXt50Encoder()
    return FPNCatSegmentationModel(encoder, num_classes=num_classes, fpn_channels=128, dropout=dropout)


def seresnext101_fpncat256(num_classes=1, dropout=0.0):
    encoder = E.SEResNeXt101Encoder()
    return FPNCatSegmentationModel(encoder, num_classes=num_classes, fpn_channels=256, dropout=dropout)


def seresnext101_rfpncat256(num_classes=1, dropout=0.0):
    encoder = E.SEResNeXt101Encoder()
    return RFPNCatSegmentationModel(encoder, num_classes=num_classes, fpn_channels=256, dropout=dropout)


def seresnext101_fpnsum256(num_classes=1, dropout=0.0):
    encoder = E.SEResNeXt101Encoder()
    return FPNSumSegmentationModel(encoder, num_classes=num_classes, fpn_channels=256, dropout=dropout)


def resnet152_fpncat256(num_classes=1, dropout=0.0):
    encoder = E.Resnet152Encoder()
    return FPNCatSegmentationModel(encoder, num_classes=num_classes, fpn_channels=256, dropout=dropout)


def effnetB4_fpncat128(num_classes=1, dropout=0.0):
    encoder = E.EfficientNetB4Encoder(abn_params={"activation": "swish"})
    return FPNCatSegmentationModel(encoder, num_classes=num_classes, fpn_channels=128, dropout=dropout)
