from functools import partial

import torch
from pytorch_toolbelt.inference.functional import pad_image_tensor, unpad_image_tensor
from pytorch_toolbelt.modules import encoders as E
from pytorch_toolbelt.modules import decoders as D
from pytorch_toolbelt.modules.abn import ACT_SELU
from pytorch_toolbelt.modules.fpn import FPNFuse, FPNBottleneckBlockBN
from pytorch_toolbelt.utils.torch_utils import count_parameters
from torch import nn
from torch.nn import functional as F

from pytorch_toolbelt.modules.unet import UnetCentralBlock, UnetDecoderBlock, UnetEncoderBlock


class FPNSegmentationModel(nn.Module):
    def __init__(self, encoder: E.EncoderModule, decoder: D.DecoderModule, num_classes: int, dropout=0.25):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.fpn_fuse = FPNFuse()
        self.dropout = nn.Dropout2d(dropout, inplace=True)

        # Final Classifier
        output_features = sum(self.decoder.output_filters)

        self.logits = nn.Conv2d(output_features, num_classes, kernel_size=1)

    def forward(self, x):
        x, pad = pad_image_tensor(x, 32)

        enc_features = self.encoder(x)
        dec_features = self.decoder(enc_features)

        features = self.fpn_fuse(dec_features)
        features = self.dropout(features)

        logits = self.logits(features)
        logits = F.interpolate(logits, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
        logits = unpad_image_tensor(logits, pad)

        return logits

    def set_encoder_training_enabled(self, enabled):
        self.encoder.set_trainable(enabled)


class FPNSegmentationModelV2(nn.Module):
    def __init__(self, encoder: E.EncoderModule, decoder: D.DecoderModule, num_classes: int, dropout=0.25):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.fpn_fuse = FPNFuse()
        self.dropout = nn.Dropout2d(dropout, inplace=True)

        # Final Classifier
        output_features = sum(self.decoder.output_filters)

        self.smooth1 = DoubleConvReluResidual(output_features, output_features // 2)
        self.smooth2 = DoubleConvReluResidual(output_features // 2, output_features // 4)

        self.logits = nn.Conv2d(output_features // 4, num_classes, kernel_size=1)

    def forward(self, x):
        x, pad = pad_image_tensor(x, 32)

        enc_features = self.encoder(x)
        dec_features = self.decoder(enc_features)

        features = self.fpn_fuse(dec_features)
        features = self.dropout(features)

        features = F.interpolate(features, scale_factor=2, mode='bilinear', align_corners=False)
        features = self.smooth1(features)

        features = F.interpolate(features, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        features = self.smooth2(features)

        logits = self.logits(features)
        logits = unpad_image_tensor(logits, pad)

        return logits

    def set_encoder_training_enabled(self, enabled):
        self.encoder.set_trainable(enabled)


class DoubleConvRelu(nn.Module):
    def __init__(self, in_dec_filters: int, out_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dec_filters, out_filters, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        return x


class DoubleConvReluResidual(nn.Module):
    def __init__(self, in_dec_filters: int, out_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dec_filters, out_filters, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=3, padding=1, stride=1)
        self.residual = nn.Conv2d(in_dec_filters, out_filters, kernel_size=1)

    def forward(self, x):
        residual = self.residual(x)

        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        return x + residual


class ConvBNRelu(nn.Module):
    def __init__(self, in_dec_filters: int, out_filters: int):
        super().__init__()
        self.conv = nn.Conv2d(in_dec_filters, out_filters, kernel_size=3, padding=1, stride=1)
        self.bn = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        return x


def fpn128_resnet34(num_classes=1, num_channels=3):
    assert num_channels == 3
    encoder = E.Resnet34Encoder()
    decoder = D.FPNDecoder(features=encoder.output_filters,
                           prediction_block=DoubleConvRelu,
                           bottleneck=FPNBottleneckBlockBN,
                           fpn_features=128)

    return FPNSegmentationModel(encoder, decoder, num_classes)


def fpn128_resnet34_v2(num_classes=1, num_channels=3):
    assert num_channels == 3
    encoder = E.Resnet34Encoder()
    decoder = D.FPNDecoder(features=encoder.output_filters,
                           prediction_block=DoubleConvRelu,
                           bottleneck=FPNBottleneckBlockBN,
                           fpn_features=128)

    return FPNSegmentationModelV2(encoder, decoder, num_classes)


def fpn128_resnext50(num_classes=1, num_channels=3):
    assert num_channels == 3
    encoder = E.SEResNeXt50Encoder()
    decoder = D.FPNDecoder(features=encoder.output_filters,
                           prediction_block=DoubleConvRelu,
                           bottleneck=FPNBottleneckBlockBN,
                           fpn_features=128)

    return FPNSegmentationModel(encoder, decoder, num_classes)


def fpn128_resnext50_v2(num_classes=1, num_channels=3):
    assert num_channels == 3
    encoder = E.SEResNeXt50Encoder()
    decoder = D.FPNDecoder(features=encoder.output_filters,
                           prediction_block=DoubleConvRelu,
                           bottleneck=FPNBottleneckBlockBN,
                           fpn_features=128)

    return FPNSegmentationModelV2(encoder, decoder, num_classes)


def fpn256_resnext50(num_classes=1, num_channels=3):
    assert num_channels == 3
    encoder = E.SEResNeXt50Encoder()
    decoder = D.FPNDecoder(features=encoder.output_filters,
                           prediction_block=DoubleConvRelu,
                           bottleneck=FPNBottleneckBlockBN,
                           fpn_features=256)

    return FPNSegmentationModel(encoder, decoder, num_classes)


def fpn_senet154(num_classes=1, num_channels=3):
    assert num_channels == 3
    encoder = E.SENet154Encoder()
    decoder = D.FPNDecoder(features=encoder.output_filters,
                           bottleneck=FPNBottleneckBlockBN,
                           prediction_block=UnetEncoderBlock,
                           fpn_features=256,
                           prediction_features=[128, 256, 512, 768])

    return FPNSegmentationModel(encoder, decoder, num_classes)
