from collections import OrderedDict
from typing import Union, List, Dict

from pytorch_toolbelt.modules import encoders as E
from pytorch_toolbelt.modules import decoders as D, conv1x1
from pytorch_toolbelt.modules.decoders.can import CANDecoder
from torch import nn, Tensor
from torch.nn import functional as F

from inria.dataset import OUTPUT_MASK_KEY

__all__ = ["CANSegmentationModel", "seresnext50_can"]


class CANSegmentationModel(nn.Module):
    def __init__(
        self, encoder: E.EncoderModule, features=256, num_classes: int = 1, dropout=0.25, full_size_mask=True
    ):
        super().__init__()
        self.encoder = encoder

        self.decoder = CANDecoder(encoder.channels, out_channels=features)

        self.mask = nn.Sequential(
            OrderedDict([("drop", nn.Dropout2d(dropout)), ("conv", conv1x1(features, num_classes))])
        )

        self.full_size_mask = full_size_mask

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x_size = x.size()
        x = self.encoder(x)
        x = self.decoder(x)

        # Decode mask
        mask = self.mask(x[0])

        if self.full_size_mask:
            mask = F.interpolate(mask, size=x_size[2:], mode="bilinear", align_corners=False)

        output = {OUTPUT_MASK_KEY: mask}
        return output


def seresnext50_can(input_channels=3, num_classes=1, dropout=0.0, pretrained=True):
    encoder = E.Resnet50Encoder(pretrained=pretrained, layers=[1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return CANSegmentationModel(encoder, num_classes=num_classes, features=256, dropout=dropout)
