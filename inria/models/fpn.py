import inspect
from functools import partial
from types import LambdaType

from pytorch_toolbelt.inference.functional import pad_image_tensor, unpad_image_tensor
from pytorch_toolbelt.modules import decoders as D
from pytorch_toolbelt.modules import encoders as E
from pytorch_toolbelt.modules.activations import Swish
from pytorch_toolbelt.modules.dsconv import DepthwiseSeparableConv2d
from pytorch_toolbelt.modules.fpn import *
from torch import nn
from torch.nn import functional as F


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


class DoubleConvBNRelu(nn.Module):
    def __init__(self, in_dec_filters: int, out_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dec_filters, out_filters, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_dec_filters)

        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        return x


class UpsampleSmooth(nn.Module):
    def __init__(self, filters: int, upsample_scale=2, mode='bilinear', align_corners=True):
        super().__init__()
        self.interpolation_mode = mode
        self.upsample_scale = upsample_scale
        self.align_corners = align_corners
        self.conv = nn.Conv2d(filters, filters,
                              kernel_size=3,
                              padding=1)

    def forward(self, x):
        x = F.interpolate(x,
                          scale_factor=self.upsample_scale,
                          mode=self.interpolation_mode,
                          align_corners=self.align_corners)

        x = F.elu(self.conv(x), inplace=True) + x
        return x


class DoubleConvReluResidual(nn.Module):
    def __init__(self, in_dec_filters: int, out_filters: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(out_filters)
        self.conv1 = DepthwiseSeparableConv2d(in_dec_filters, out_filters, kernel_size=3, padding=1, stride=1)
        self.conv2 = DepthwiseSeparableConv2d(out_filters, out_filters, kernel_size=3, padding=1, stride=1)
        self.residual = nn.Conv2d(in_dec_filters, out_filters, kernel_size=1)

    def forward(self, x):
        residual = self.residual(x)
        residual = self.bn(residual)

        x = self.conv1(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.conv2(x)
        x = F.leaky_relu(x, inplace=True)
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


class FPNSegmentationModel(nn.Module):
    def __init__(self, encoder: E.EncoderModule, num_classes: int, fpn_features=128, dropout=0.25):
        super().__init__()

        decoder = D.FPNDecoder(features=encoder.output_filters,
                               prediction_block=DoubleConvRelu,
                               bottleneck=FPNBottleneckBlockBN,
                               fpn_features=fpn_features,
                               prediction_features=fpn_features)

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


def reset34_fpn(num_classes=1, fpn_features=128):
    encoder = E.Resnet34Encoder()
    return FPNSegmentationModel(encoder, num_classes, fpn_features)


def reset101_fpn(num_classes=1, fpn_features=256):
    encoder = E.Resnet101Encoder()
    return FPNSegmentationModel(encoder, num_classes, fpn_features)


def effnetB4_fpn(num_classes=1, fpn_features=128):
    encoder = E.EfficientNetB4Encoder()
    return FPNSegmentationModel(encoder, num_classes, fpn_features)


def effnetB7_fpn(num_classes=1, fpn_features=256):
    encoder = E.EfficientNetB7Encoder()
    return FPNSegmentationModel(encoder, num_classes, fpn_features)
