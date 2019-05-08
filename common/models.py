import inspect
from functools import partial
from types import LambdaType

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

        self.logits = nn.Conv2d(output_features, num_classes, kernel_size=1)
        self.edges = nn.Conv2d(output_features, num_classes, kernel_size=1)

    def forward(self, x):
        x, pad = pad_image_tensor(x, 32)

        enc_features = self.encoder(x)
        dec_features = self.decoder(enc_features)

        features = self.fpn_fuse(dec_features)
        features = self.dropout(features)

        edges = self.edges(features)
        logits = self.logits(features) - F.relu(edges)

        logits = F.interpolate(logits, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
        logits = unpad_image_tensor(logits, pad)

        edges = F.interpolate(edges, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
        edges = unpad_image_tensor(edges, pad)

        return {"logits": logits, "edge": edges}

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


def fpn_v1(encoder, num_classes=1, num_channels=3, fpn_features=128):
    assert num_channels == 3
    if inspect.isclass(encoder):
        encoder = encoder()
    elif isinstance(encoder, LambdaType):
        encoder = encoder()

    assert isinstance(encoder, E.EncoderModule)

    decoder = D.FPNDecoder(features=encoder.output_filters,
                           prediction_block=DoubleConvRelu,
                           bottleneck=FPNBottleneckBlockBN,
                           fpn_features=fpn_features)

    return FPNSegmentationModel(encoder, decoder, num_classes)


def fpn_v2(encoder, num_classes=1, num_channels=3, fpn_features=128):
    assert num_channels == 3

    if inspect.isclass(encoder):
        encoder = encoder()
    elif isinstance(encoder, LambdaType):
        encoder = encoder()

    assert isinstance(encoder, E.EncoderModule)
    decoder = D.FPNDecoder(features=encoder.output_filters,
                           prediction_block=DoubleConvRelu,
                           bottleneck=FPNBottleneckBlockBN,
                           fpn_features=fpn_features)

    return FPNSegmentationModelV2(encoder, decoder, num_classes)
