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
            output_channels=num_classes,
            dsv_channels=num_classes,
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
            output_channels=num_classes,
            dsv_channels=num_classes,
            fpn_channels=fpn_channels,
            abn_block=abn_block,
            dropout=dropout,
        )

        self.full_size_mask = full_size_mask

    def forward(self, x):
        enc_features = self.encoder(x)

        # Decode mask
        mask, dsv = self.decoder(enc_features)

        if self.full_size_mask:
            mask = F.interpolate(mask, size=x.size()[2:], mode="bilinear", align_corners=False)

        output = {
            OUTPUT_MASK_KEY: mask,
            OUTPUT_MASK_4_KEY: dsv[0],
            OUTPUT_MASK_8_KEY: dsv[1],
            OUTPUT_MASK_16_KEY: dsv[2],
            OUTPUT_MASK_32_KEY: dsv[3],
        }

        return output


class DualFPNCatSegmentationModel(nn.Module):
    def __init__(
        self,
        encoder: EncoderModule,
        localization_classes=1,
        damage_classes=5,
        dropout=0.25,
        abn_block=ABN,
        fpn_channels=256,
        full_size_mask=True,
    ):
        super().__init__()
        self.encoder = encoder

        self.pre_decoder = FPNCatDecoder(
            feature_maps=encoder.output_filters,
            output_channels=localization_classes,
            fpn_channels=fpn_channels,
            abn_block=abn_block,
            dropout=dropout,
        )

        self.post_decoder = FPNCatDecoder(
            feature_maps=[f + f for f in encoder.output_filters],
            output_channels=damage_classes,
            fpn_channels=fpn_channels,
            abn_block=abn_block,
            dropout=dropout,
        )

        self.full_size_mask = full_size_mask

    def forward(self, image_pre=None, image_post=None):
        pre_features = self.encoder(image_pre)
        post_features = self.encoder(image_post)

        # Decode mask
        pre_mask = self.pre_decoder(pre_features)

        features = [torch.cat((pre, post), dim=1) for (pre, post) in zip(pre_features, post_features)]
        post_mask = self.post_decoder(features)

        if self.full_size_mask:
            pre_mask = F.interpolate(pre_mask, size=image_pre.size()[2:], mode="bilinear", align_corners=False)
            post_mask = F.interpolate(post_mask, size=image_post.size()[2:], mode="bilinear", align_corners=False)

        output = {OUTPUT_MASK_PRE_KEY: pre_mask, OUTPUT_MASK_POST_KEY: post_mask}

        return output


def resnet34_fpncat128(num_classes=5, dropout=0.0, pretrained=True):
    encoder = E.Resnet34Encoder(pretrained=pretrained)
    return FPNCatSegmentationModel(encoder, num_classes=num_classes, fpn_channels=128, dropout=dropout)


def seresnext50_fpncat128(num_classes=5, dropout=0.0, pretrained=True):
    encoder = E.SEResNeXt50Encoder(pretrained=pretrained)
    return FPNCatSegmentationModel(encoder, num_classes=num_classes, fpn_channels=128, dropout=dropout)


def seresnext101_fpncat256(num_classes=5, dropout=0.0, pretrained=True):
    encoder = E.SEResNeXt101Encoder(pretrained=pretrained)
    return FPNCatSegmentationModel(encoder, num_classes=num_classes, fpn_channels=256, dropout=dropout)


def seresnext101_fpnsum256(num_classes=5, dropout=0.0, pretrained=True):
    encoder = E.SEResNeXt101Encoder(pretrained=pretrained)
    return FPNSumSegmentationModel(encoder, num_classes=num_classes, fpn_channels=256, dropout=dropout)


def resnet152_fpncat256(num_classes=5, dropout=0.0, pretrained=True):
    encoder = E.Resnet152Encoder(pretrained=pretrained)
    return FPNCatSegmentationModel(encoder, num_classes=num_classes, fpn_channels=256, dropout=dropout)


def effnetB4_fpncat128(num_classes=5, dropout=0.0, pretrained=True):
    encoder = E.EfficientNetB4Encoder(abn_params={"activation": "swish"}, pretrained=pretrained)
    return FPNCatSegmentationModel(encoder, num_classes=num_classes, fpn_channels=128, dropout=dropout)
