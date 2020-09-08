import math
import warnings
from collections import OrderedDict
from typing import List, Union, Type
import torch
from pytorch_toolbelt.modules.encoders import _take
from torch import nn
from timm.models import tresnet_m, tresnet_l, tresnet_xl, skresnext50_32x4d, skresnet18
from timm.models.resnet import swsl_resnext101_32x8d
from timm.models.efficientnet import (
    tf_efficientnet_b2_ns,
    tf_efficientnet_b4_ns,
    tf_efficientnet_b6_ns,
    tf_efficientnet_b0_ns,
    tf_efficientnet_b7_ns,
    mixnet_xl,
    efficientnet_b4,
    tf_efficientnet_b3_ns,
    tf_efficientnet_b5_ns,
)
from timm.models import hrnet
from timm.models.layers import Swish, Mish, Conv2dSame
from pytorch_toolbelt.modules import EncoderModule, make_n_channel_input
from pytorch_toolbelt.utils import count_parameters

__all__ = [
    "B0Encoder",
    "B2Encoder",
    "B3Encoder",
    "B4Encoder",
    "B5Encoder",
    "B6Encoder",
    "B7Encoder",
    "MixNetXLEncoder",
    "HRNetW32Encoder",
    "HRNetW18Encoder",
    "HRNetW48Encoder",
    "NoStrideB2Encoder",
    "NoStrideB4Encoder",
    "NoStrideB6Encoder",
    "SKResNet18Encoder",
    "SKResNeXt50Encoder",
    "SWSLResNeXt101Encoder",
    "TResNetMEncoder",
]


def make_n_channel_input_conv2d_same(conv: nn.Conv2d, in_channels: int, mode="auto"):
    assert isinstance(conv, nn.Conv2d)
    if conv.in_channels == in_channels:
        warnings.warn("make_n_channel_input call is spurious")
        return conv

    new_conv = Conv2dSame(
        in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
    )

    w = conv.weight
    if in_channels > conv.in_channels:
        n = math.ceil(in_channels / float(conv.in_channels))
        w = torch.cat([w] * n, dim=1)
        w = w[:, :in_channels, ...]
        new_conv.weight = nn.Parameter(w, requires_grad=True)
    else:
        w = w[:, 0:in_channels, ...]
        new_conv.weight = nn.Parameter(w, requires_grad=True)

    return new_conv


class B0Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=[1, 2, 3, 4]):
        encoder = tf_efficientnet_b0_ns(pretrained=pretrained, features_only=True, drop_path_rate=0.05)
        super().__init__([16, 24, 40, 112, 320], [2, 4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        features = self.encoder(x)
        return _take(features, self._layers)

    def change_input_channels(self, input_channels: int, mode="auto"):
        self.encoder.conv_stem = make_n_channel_input_conv2d_same(self.encoder.conv_stem, input_channels, mode)
        return self


class B2Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=[1, 2, 3, 4]):
        encoder = tf_efficientnet_b2_ns(pretrained=pretrained, features_only=True, drop_path_rate=0.1)
        super().__init__([16, 24, 48, 120, 352], [2, 4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        features = self.encoder(x)
        return _take(features, self._layers)

    def change_input_channels(self, input_channels: int, mode="auto"):
        self.encoder.conv_stem = make_n_channel_input_conv2d_same(self.encoder.conv_stem, input_channels, mode)
        return self


class B3Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=[1, 2, 3, 4], act_layer=Swish):
        encoder = tf_efficientnet_b3_ns(
            pretrained=pretrained, features_only=True, act_layer=act_layer, drop_path_rate=0.1
        )
        super().__init__([24, 32, 48, 136, 384], [2, 4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        features = self.encoder(x)
        return _take(features, self._layers)

    def change_input_channels(self, input_channels: int, mode="auto"):
        self.encoder.conv_stem = make_n_channel_input_conv2d_same(self.encoder.conv_stem, input_channels, mode)
        return self


class B4Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=[1, 2, 3, 4], act_layer=Swish):
        encoder = tf_efficientnet_b4_ns(
            pretrained=pretrained, features_only=True, act_layer=act_layer, drop_path_rate=0.2
        )
        super().__init__([24, 32, 56, 160, 448], [2, 4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        features = self.encoder(x)
        return _take(features, self._layers)

    def change_input_channels(self, input_channels: int, mode="auto"):
        self.encoder.conv_stem = make_n_channel_input_conv2d_same(self.encoder.conv_stem, input_channels, mode)
        return self


class B5Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=[1, 2, 3, 4], act_layer=Swish):
        encoder = tf_efficientnet_b5_ns(
            pretrained=pretrained, features_only=True, act_layer=act_layer, drop_path_rate=0.2
        )
        super().__init__([24, 40, 64, 176, 512], [2, 4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        features = self.encoder(x)
        return _take(features, self._layers)

    def change_input_channels(self, input_channels: int, mode="auto"):
        self.encoder.conv_stem = make_n_channel_input_conv2d_same(self.encoder.conv_stem, input_channels, mode)
        return self


class B6Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=[1, 2, 3, 4], act_layer=Swish):
        encoder = tf_efficientnet_b6_ns(
            pretrained=pretrained, features_only=True, act_layer=act_layer, drop_path_rate=0.2
        )
        super().__init__([32, 40, 72, 200, 576], [2, 4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        features = self.encoder(x)
        return _take(features, self._layers)

    def change_input_channels(self, input_channels: int, mode="auto"):
        self.encoder.conv_stem = make_n_channel_input_conv2d_same(self.encoder.conv_stem, input_channels, mode)
        return self


class B7Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=[1, 2, 3, 4], act_layer=Swish):
        encoder = tf_efficientnet_b7_ns(
            pretrained=pretrained, features_only=True, act_layer=act_layer, drop_path_rate=0.2
        )
        super().__init__([32, 48, 80, 224, 640], [2, 4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        features = self.encoder(x)
        return _take(features, self._layers)

    def change_input_channels(self, input_channels: int, mode="auto"):
        self.encoder.conv_stem = make_n_channel_input_conv2d_same(self.encoder.conv_stem, input_channels, mode)
        return self


class MixNetXLEncoder(EncoderModule):
    def __init__(self, pretrained=True, layers=[1, 2, 3, 4], act_layer=Swish):
        encoder = mixnet_xl(pretrained=pretrained, features_only=True, act_layer=act_layer, drop_path_rate=0.2)
        super().__init__([40, 48, 64, 192, 320], [2, 4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        features = self.encoder(x)
        return _take(features, self._layers)

    def change_input_channels(self, input_channels: int, mode="auto"):
        self.encoder.conv_stem = make_n_channel_input_conv2d_same(self.encoder.conv_stem, input_channels, mode)
        return self


class NoStrideB2Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=[1, 2, 3, 4]):
        encoder = tf_efficientnet_b2_ns(pretrained=pretrained, features_only=True, drop_path_rate=0.1)
        # print(encoder)
        encoder.blocks[5][0].conv_dw.stride = (1, 1)
        encoder.blocks[5][0].conv_dw.dilation = (2, 2)

        encoder.blocks[3][0].conv_dw.stride = (1, 1)
        encoder.blocks[3][0].conv_dw.dilation = (2, 2)

        super().__init__([16, 24, 48, 120, 352], [2, 4, 8, 8, 8], layers)
        self.encoder = encoder

    def forward(self, x):
        features = self.encoder(x)
        return _take(features, self._layers)


class NoStrideB4Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=[1, 2, 3, 4]):
        encoder = tf_efficientnet_b4_ns(pretrained=pretrained, features_only=True, drop_path_rate=0.2)
        # print(encoder)
        encoder.blocks[5][0].conv_dw.stride = (1, 1)
        encoder.blocks[5][0].conv_dw.dilation = (2, 2)

        encoder.blocks[3][0].conv_dw.stride = (1, 1)
        encoder.blocks[3][0].conv_dw.dilation = (2, 2)

        super().__init__([24, 32, 56, 160, 448], [2, 4, 8, 8, 8], layers)
        self.encoder = encoder

    def forward(self, x):
        features = self.encoder(x)
        return _take(features, self._layers)


class NoStrideB6Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=[1, 2, 3, 4]):
        encoder = tf_efficientnet_b6_ns(pretrained=pretrained, features_only=True, drop_path_rate=0.2)
        # print(encoder)
        encoder.blocks[5][0].conv_dw.stride = (1, 1)
        encoder.blocks[5][0].conv_dw.dilation = (2, 2)

        encoder.blocks[3][0].conv_dw.stride = (1, 1)
        encoder.blocks[3][0].conv_dw.dilation = (2, 2)

        super().__init__([32, 40, 72, 200, 576], [2, 4, 8, 8, 8], layers)
        self.encoder = encoder

    def forward(self, x):
        features = self.encoder(x)
        return _take(features, self._layers)


class TResNetMEncoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        if layers is None:
            layers = [1, 2, 3, 4]
        encoder = tresnet_m(pretrained=pretrained)

        super().__init__([64, 64, 128, 1024, 2048], [4, 4, 8, 16, 32], layers)
        self.stem = nn.Sequential(encoder.body.SpaceToDepth, encoder.body.conv1)

        self.layer1 = encoder.body.layer1
        self.layer2 = encoder.body.layer2
        self.layer3 = encoder.body.layer3
        self.layer4 = encoder.body.layer4

    @property
    def encoder_layers(self) -> List[nn.Module]:
        return [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]


class SKResNet18Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None, no_first_max_pool=False):
        if layers is None:
            layers = [1, 2, 3, 4]
        encoder = skresnet18(pretrained=pretrained, features_only=True)
        super().__init__([64, 64, 128, 256, 512], [2, 4, 8, 16, 32], layers)
        self.stem = nn.Sequential(
            OrderedDict([("conv1", encoder.conv1), ("bn1", encoder.bn1), ("act1", encoder.act1)])
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2) if no_first_max_pool else encoder.maxpool,
            encoder.layer1,
        )
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

    @property
    def encoder_layers(self) -> List[nn.Module]:
        return [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]

    def change_input_channels(self, input_channels: int, mode="auto"):
        self.stem.conv1 = make_n_channel_input(self.stem.conv1, input_channels, mode)
        return self


class SKResNeXt50Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        if layers is None:
            layers = [1, 2, 3, 4]
        encoder = skresnext50_32x4d(pretrained=pretrained)
        super().__init__([64, 256, 512, 1024, 2048], [2, 4, 8, 16, 32], layers)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.act1)

        self.layer1 = nn.Sequential(encoder.maxpool, encoder.layer1)
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

    @property
    def encoder_layers(self) -> List[nn.Module]:
        return [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]


class SWSLResNeXt101Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        if layers is None:
            layers = [1, 2, 3, 4]
        encoder = swsl_resnext101_32x8d(pretrained=pretrained)
        super().__init__([64, 256, 512, 1024, 2048], [2, 4, 8, 16, 32], layers)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.act1)

        self.layer1 = nn.Sequential(encoder.maxpool, encoder.layer1)
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

    @property
    def encoder_layers(self) -> List[nn.Module]:
        return [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]


class HRNetW48Encoder(EncoderModule):
    def __init__(self, pretrained=True):
        encoder = hrnet.hrnet_w48(pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4))
        super().__init__([128, 256, 512, 1024], [4, 8, 16, 32], [0, 1, 2, 3])
        self.encoder = encoder

    def forward(self, x):
        y = self.encoder.forward(x)
        return y

    def change_input_channels(self, input_channels: int, mode="auto"):
        self.encoder.conv1 = make_n_channel_input(self.encoder.conv1, input_channels, mode)
        return self


class HRNetW32Encoder(EncoderModule):
    def __init__(self, pretrained=True):
        encoder = hrnet.hrnet_w32(pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4))
        super().__init__([128, 256, 512, 1024], [4, 8, 16, 32], [0, 1, 2, 3])
        self.encoder = encoder

    def forward(self, x):
        y = self.encoder.forward(x)
        return y

    def change_input_channels(self, input_channels: int, mode="auto"):
        self.encoder.conv1 = make_n_channel_input(self.encoder.conv1, input_channels, mode)
        return self


class HRNetW18Encoder(EncoderModule):
    def __init__(self, pretrained=True):
        encoder = hrnet.hrnet_w18(pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4))
        super().__init__([128, 256, 512, 1024], [4, 8, 16, 32], [0, 1, 2, 3])
        self.encoder = encoder

    def forward(self, x):
        y = self.encoder.forward(x)
        return y

    def change_input_channels(self, input_channels: int, mode="auto"):
        self.encoder.conv1 = make_n_channel_input(self.encoder.conv1, input_channels, mode)
        return self


if __name__ == "__main__":
    encoder = HRNetW18Encoder()
    print(count_parameters(encoder))
    print(encoder.channels, encoder.strides)

    x = torch.randn((2, 3, 512, 512))
    y = encoder(x)
    for out in y:
        print(out.size())
