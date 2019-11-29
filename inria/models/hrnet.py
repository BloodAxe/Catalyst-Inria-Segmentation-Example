from pytorch_toolbelt.modules import encoders as E
from pytorch_toolbelt.modules.decoders import HRNetDecoder
from pytorch_toolbelt.modules.encoders import EncoderModule
from torch import nn
from torch.nn import functional as F

from ..dataset import (
    OUTPUT_MASK_KEY,
)

__all__ = ["HRNetSegmentationModel", "hrnet18"]


class HRNetSegmentationModel(nn.Module):
    def __init__(self, encoder: EncoderModule, num_classes: int, dropout=0.25, full_size_mask=True):
        super().__init__()
        self.encoder = encoder

        self.decoder = HRNetDecoder(feature_maps=encoder.output_filters, output_channels=num_classes, dropout=dropout)

        self.full_size_mask = full_size_mask

    def forward(self, x):
        enc_features = self.encoder(x)

        # Decode mask
        mask = self.decoder(enc_features)

        if self.full_size_mask:
            mask = F.interpolate(mask, size=x.size()[2:], mode="bilinear", align_corners=False)

        output = {OUTPUT_MASK_KEY: mask}

        return output


def hrnet18(num_classes=5, dropout=0.0, pretrained=True):
    encoder = E.HRNetV2Encoder18(pretrained=pretrained)
    return HRNetSegmentationModel(encoder, num_classes=num_classes, dropout=dropout)


def hrnet34(num_classes=5, dropout=0.0, pretrained=True):
    encoder = E.HRNetV2Encoder34(pretrained=pretrained)
    return HRNetSegmentationModel(encoder, num_classes=num_classes, dropout=dropout)


def hrnet48(num_classes=5, dropout=0.0, pretrained=True):
    encoder = E.HRNetV2Encoder48(pretrained=pretrained)
    return HRNetSegmentationModel(encoder, num_classes=num_classes, dropout=dropout)
