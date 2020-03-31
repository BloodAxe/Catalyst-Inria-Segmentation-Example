from pytorch_toolbelt.modules.encoders import EncoderModule, StackedHGEncoder
from pytorch_toolbelt.modules.encoders.hourglass import StackedSupervisedHGEncoder
from torch import nn
from torch.nn import PixelShuffle

from inria.dataset import OUTPUT_MASK_KEY, OUTPUT_MASK_4_KEY
import torch.nn.functional as F


class HGSegmentationDecoderNaked(nn.Module):
    def __init__(self, input_channels: int, stride: int, mask_channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(input_channels, mask_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class HGSegmentationDecoder(nn.Module):
    def __init__(self, input_channels: int, stride: int, mask_channels: int):
        super().__init__()
        self.expand = nn.Conv2d(input_channels, input_channels, kernel_size=1)
        self.up = PixelShuffle(upscale_factor=stride)

        mid_channels = input_channels // (2 ** stride)
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act1 = nn.ReLU(True)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.act2 = nn.ReLU(True)

        self.final = nn.Conv2d(mid_channels, mask_channels, kernel_size=1)

    def forward(self, x):
        x = self.expand(x)
        x = self.up(x)

        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))

        x = self.final(x)
        return x


class HGSegmentationModel(nn.Module):
    def __init__(self, encoder: EncoderModule, num_classes: int, full_size_mask=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = HGSegmentationDecoder(encoder.output_filters[-1], encoder.output_strides[-1], num_classes)
        self.full_size_mask = full_size_mask

    def forward(self, x):
        features = self.encoder(x)

        # Decode mask
        mask = self.decoder(features[-1])

        output = {OUTPUT_MASK_KEY: mask}
        return output


class SupervisedHGSegmentationModel(nn.Module):
    def __init__(self, encoder: EncoderModule, num_classes: int, full_size_mask=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = HGSegmentationDecoder(encoder.output_filters[-1], encoder.output_strides[-1], num_classes)
        self.full_size_mask = full_size_mask

    def forward(self, x):
        features, supervision = self.encoder(x)

        # Decode mask
        mask = self.decoder(features[-1])

        output = {OUTPUT_MASK_KEY: mask}
        for i, sup in enumerate(supervision):
            output[OUTPUT_MASK_4_KEY + "_after_hg_" + str(i)] = sup

        return output


def hg4(num_classes=1, dropout=0, pretrained=False):
    encoder = StackedHGEncoder(stack_level=4)
    return HGSegmentationModel(encoder, num_classes=num_classes)


def shg4(num_classes=1, dropout=0, pretrained=False):
    encoder = StackedSupervisedHGEncoder(input_channels=3, stack_level=4, supervision_channels=num_classes)
    return SupervisedHGSegmentationModel(encoder, num_classes=num_classes)


def hg8(num_classes=1, dropout=0, pretrained=False):
    encoder = StackedHGEncoder(stack_level=8)
    return HGSegmentationModel(encoder, num_classes=num_classes)
