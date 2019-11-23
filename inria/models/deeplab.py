from pytorch_toolbelt.inference.functional import pad_image_tensor, unpad_image_tensor
from pytorch_toolbelt.modules import ABN
from pytorch_toolbelt.modules import encoders as E
from pytorch_toolbelt.modules.decoders import DeeplabV3Decoder
from pytorch_toolbelt.modules.encoders import EncoderModule
from torch import nn
from torch.nn import functional as F

from ..dataset import OUTPUT_MASK_4_KEY, OUTPUT_MASK_8_KEY, OUTPUT_MASK_16_KEY, OUTPUT_MASK_32_KEY, OUTPUT_MASK_KEY

__all__ = [
    "DeeplabV3SegmentationModel",
    "resnet34_deeplab128",
    "seresnext101_deeplab256"
]


class DeeplabV3SegmentationModel(nn.Module):
    def __init__(
        self,
        encoder: EncoderModule,
        num_classes: int,
        dropout=0.25,
        abn_block=ABN,
        high_level_bottleneck=256,
        low_level_bottleneck=32,
        full_size_mask=True,
    ):
        super().__init__()
        self.encoder = encoder

        self.decoder = DeeplabV3Decoder(
            feature_maps=encoder.output_filters,
            output_stride=encoder.output_strides[-1],
            num_classes=num_classes,
            high_level_bottleneck=high_level_bottleneck,
            low_level_bottleneck=low_level_bottleneck,
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
            OUTPUT_MASK_32_KEY: dsv,
        }

        return output


def resnet34_deeplab128(num_classes=1, dropout=0.0):
    encoder = E.Resnet34Encoder()
    return DeeplabV3SegmentationModel(encoder, num_classes=num_classes, high_level_bottleneck=128, dropout=dropout)


def seresnext101_deeplab256(num_classes=1, dropout=0.0):
    encoder = E.SEResNeXt101Encoder()
    return DeeplabV3SegmentationModel(encoder, num_classes=num_classes, high_level_bottleneck=256, dropout=dropout)
