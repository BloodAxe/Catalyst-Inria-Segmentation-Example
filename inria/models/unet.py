from pytorch_toolbelt.inference.functional import pad_image_tensor, unpad_image_tensor
from pytorch_toolbelt.modules import ABN
from pytorch_toolbelt.modules import encoders as E
from pytorch_toolbelt.modules.decoders import FPNSumDecoder, FPNCatDecoder, UNetDecoder, UNetDecoderV2
from pytorch_toolbelt.modules.encoders import EncoderModule
from torch import nn
from torch.nn import functional as F

from ..dataset import OUTPUT_MASK_4_KEY, OUTPUT_MASK_8_KEY, OUTPUT_MASK_16_KEY, OUTPUT_MASK_32_KEY, OUTPUT_MASK_KEY

__all__ = ["seresnext101_unet256"]


class UnetSegmentationModel(nn.Module):
    def __init__(
        self,
        encoder: EncoderModule,
        num_classes: int,
        dropout=0.25,
        abn_block=ABN,
        unet_channels=32,
        full_size_mask=True,
    ):
        super().__init__()
        self.encoder = encoder

        self.decoder = UNetDecoder(
            feature_maps=encoder.output_filters, decoder_features=unet_channels, mask_channels=num_classes
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
            OUTPUT_MASK_4_KEY: dsv[0],
            OUTPUT_MASK_8_KEY: dsv[1],
            OUTPUT_MASK_16_KEY: dsv[2],
            OUTPUT_MASK_32_KEY: dsv[3],
        }

        return output


def seresnext101_unet64(num_classes=1, dropout=0.0):
    encoder = E.SEResNeXt101Encoder()
    return UnetSegmentationModel(encoder, num_classes=num_classes, unet_channels=64, dropout=dropout)
