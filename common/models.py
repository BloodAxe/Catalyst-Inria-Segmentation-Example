import inspect
from functools import partial
from types import LambdaType

import torch
from pytorch_toolbelt.inference.functional import pad_image_tensor, unpad_image_tensor
from pytorch_toolbelt.modules import encoders as E
from pytorch_toolbelt.modules import decoders as D
from pytorch_toolbelt.modules.activations import Swish
from pytorch_toolbelt.modules.backbone.senet import SEResNeXtBottleneck
from pytorch_toolbelt.modules.dsconv import DepthwiseSeparableConv2d
from pytorch_toolbelt.modules.scse import ChannelSpatialGate2d, ChannelSpatialGate2dV2
from pytorch_toolbelt.modules.abn import ACT_SELU
from pytorch_toolbelt.modules.fpn import *
from pytorch_toolbelt.utils.torch_utils import count_parameters
from torch import nn
from torch.nn import functional as F

from pytorch_toolbelt.modules.unet import UnetCentralBlock, UnetDecoderBlock, UnetEncoderBlock

from common.linknet import LinkNet152, LinkNet34
from common.resnet import SwishGroupnormResnet101Encoder
from common.unet import UNet


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


class FPNSegmentationModelV3(nn.Module):
    def __init__(self, encoder: E.EncoderModule, decoder: D.DecoderModule, num_classes: int, dropout=0.5):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.fpn_fuse = FPNFuse()
        self.dropout = nn.Dropout2d(dropout, inplace=True)

        # Final Classifier
        fused_features = sum(self.decoder.output_filters)
        bottleneck_features = 128

        self.bottleneck = nn.Conv2d(fused_features, bottleneck_features, kernel_size=3, padding=1)

        self.up1 = UpsampleSmooth(bottleneck_features)
        self.up2 = UpsampleSmooth(bottleneck_features)

        self.coarse_logits = nn.Conv2d(bottleneck_features, num_classes, kernel_size=1)
        self.logits = nn.Conv2d(bottleneck_features, num_classes, kernel_size=1)
        self.edges = nn.Conv2d(bottleneck_features, num_classes, kernel_size=1)

    def forward(self, x):
        x, pad = pad_image_tensor(x, 32)

        enc_features = self.encoder(x)
        dec_features = self.decoder(enc_features)

        features = self.fpn_fuse(dec_features)
        features = self.dropout(features)

        features = self.bottleneck(features)

        coarse_logits = self.coarse_logits(features)

        features = self.up1(features)
        features = self.up1(features)

        edges = self.edges(features)
        logits = self.logits(features) - F.relu(edges)

        logits = unpad_image_tensor(logits, pad)
        edges = unpad_image_tensor(edges, pad)

        return {"logits": logits, "edge": edges, "coarse_logits": coarse_logits}

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


class DoubleConvGNSwish(nn.Module):
    def __init__(self, in_dec_filters: int, out_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dec_filters, out_filters, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn1 = nn.GroupNorm(32, out_filters)
        self.act1 = Swish()

        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.GroupNorm(32, out_filters)
        self.act2 = Swish()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

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


def fpn_v1(encoder, num_classes=1, num_channels=3, bottleneck_features=256, fpn_features=128):
    assert num_channels == 3

    if inspect.isclass(encoder):
        encoder = encoder()
    elif isinstance(encoder, (LambdaType, partial)):
        encoder = encoder()

    assert isinstance(encoder, E.EncoderModule)

    decoder = D.FPNDecoder(features=encoder.output_filters,
                           prediction_block=DoubleConvRelu,
                           bottleneck=FPNBottleneckBlockBN,
                           fpn_features=bottleneck_features,
                           prediction_features=fpn_features)

    return FPNSegmentationModel(encoder, decoder, num_classes)


def fpn_v2(encoder, num_classes=1, num_channels=3, bottleneck_features=256, prediction_features=128):
    assert num_channels == 3

    if inspect.isclass(encoder):
        encoder = encoder()
    elif isinstance(encoder, (LambdaType, partial)):
        encoder = encoder()

    assert isinstance(encoder, E.EncoderModule)
    decoder = D.FPNDecoder(features=encoder.output_filters,
                           bottleneck=FPNBottleneckBlock,
                           prediction_block=DoubleConvBNRelu,
                           fpn_features=bottleneck_features,
                           prediction_features=prediction_features)

    return FPNSegmentationModelV2(encoder, decoder, num_classes)


def fpn_v3(encoder, num_classes=1, num_channels=3, bottleneck_features=256, prediction_features=128):
    assert num_channels == 3

    if inspect.isclass(encoder):
        encoder = encoder()
    elif isinstance(encoder, (LambdaType, partial)):
        encoder = encoder()

    assert isinstance(encoder, E.EncoderModule)
    decoder = D.FPNDecoder(features=encoder.output_filters,
                           bottleneck=FPNBottleneckBlock,
                           prediction_block=DoubleConvBNRelu,
                           fpn_features=bottleneck_features,
                           prediction_features=prediction_features)

    return FPNSegmentationModelV3(encoder, decoder, num_classes)


def fpn_v4(num_classes=1, num_channels=3, bottleneck_features=256, prediction_features=128):
    assert num_channels == 3

    encoder = E.Resnet101Encoder(layers=[1, 2, 3, 4])

    assert isinstance(encoder, E.EncoderModule)
    decoder = D.FPNDecoder(features=encoder.output_filters,
                           bottleneck=DoubleConvGNSwish,
                           prediction_block=DoubleConvGNSwish,
                           upsample_add_block=partial(UpsampleAdd, mode='bilinear', align_corners=True),
                           fpn_features=bottleneck_features,
                           prediction_features=prediction_features)

    return FPNSegmentationModelV2(encoder, decoder, num_classes, dropout=0.5)


def fpn_v4_swish(num_classes=1, num_channels=3, bottleneck_features=256, prediction_features=128):
    assert num_channels == 3

    encoder = SwishGroupnormResnet101Encoder(layers=[1, 2, 3, 4])

    assert isinstance(encoder, E.EncoderModule)
    decoder = D.FPNDecoder(features=encoder.output_filters,
                           bottleneck=DoubleConvGNSwish,
                           prediction_block=DoubleConvGNSwish,
                           upsample_add_block=partial(UpsampleAdd, mode='bilinear', align_corners=True),
                           fpn_features=bottleneck_features,
                           prediction_features=prediction_features)

    return FPNSegmentationModelV2(encoder, decoder, num_classes, dropout=0.5)


def fpn_v5(num_classes=1, num_channels=3):
    assert num_channels == 3

    encoder = SwishGroupnormResnet101Encoder(layers=[1, 2, 3, 4])

    assert isinstance(encoder, E.EncoderModule)
    decoder = D.FPNDecoder(features=encoder.output_filters,
                           bottleneck=DoubleConvGNSwish,
                           prediction_block=DoubleConvGNSwish,
                           upsample_add_block=partial(UpsampleAdd, mode='bilinear', align_corners=True),
                           fpn_features=[256, 512, 1024, 2048],
                           prediction_features=[64, 128, 256, 512])

    return FPNSegmentationModelV2(encoder, decoder, num_classes, dropout=0.5)


def get_model(model_name: str, image_size=None) -> nn.Module:
    registry = {
        'unet': partial(UNet, upsample=False),
        'linknet34': LinkNet34,
        'linknet152': LinkNet152,
        'fpn128_mobilenetv2': partial(fpn_v2, encoder=partial(E.MobilenetV2Encoder, layers=[2, 3, 5, 6, 7]),
                                      prediction_features=128),
        'fpn128_mobilenetv3': partial(fpn_v2, encoder=partial(E.MobilenetV3Encoder, layers=[0, 1, 2, 3, 4, 5]),
                                      prediction_features=128),
        'fpn128_resnet34': partial(fpn_v1, encoder=E.Resnet34Encoder, prediction_features=128),
        'fpn128_resnext50': partial(fpn_v1, encoder=E.SEResNeXt50Encoder, prediction_features=128),
        'fpn256_resnext50': partial(fpn_v1, encoder=E.SEResNeXt50Encoder, prediction_features=256),
        'fpn128_resnext50_v2': partial(fpn_v2, encoder=E.SEResNeXt50Encoder, prediction_features=128),

        'fpn256_resnext50_v2': partial(fpn_v2, encoder=E.SEResNeXt50Encoder, bottleneck_features=256,
                                       prediction_features=256),
        'fpn256_resnext50_v3': partial(fpn_v3, encoder=E.SEResNeXt50Encoder, bottleneck_features=256,
                                       prediction_features=256),
        'fpn256_resnext101_v3': partial(fpn_v3, encoder=E.SEResNeXt101Encoder, bottleneck_features=256,
                                        prediction_features=256),
        # 'fpn256_senet154_v2': partial(fpn_v2, encoder=E.SENet154Encoder, prediction_features=384),
        # 'fpn256_senet154_v2': partial(fpn_v2, encoder=E.SENet154Encoder, bottleneck_features=512, prediction_features=256),
        # 'ternausnetv2': partial(TernausNetV2, num_input_channels=3, num_classes=1),
        # 'fpn128_wider_resnet20': partial(fpn_v1,
        #                                  encoder=lambda _: E.WiderResnet20A2Encoder(layers=[1, 2, 3, 4, 5]),
        #                                  prediction_features=128),
        'fpn256_resnet101_v4': fpn_v4,
        'fpn256_resnet101_v4_swish': fpn_v4_swish
    }

    return registry[model_name.lower()]()


def test_fpn_v4():
    model = fpn_v4().eval()
    print(count_parameters(model))

    x = torch.rand((1, 3, 512, 512))
    y = model(x)
    print(y)


def test_fpn_v5():
    model = fpn_v5().eval()
    print(count_parameters(model))

    x = torch.rand((1, 3, 512, 512))
    y = model(x)
    print(y)
