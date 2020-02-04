import albumentations as A
import cv2

__all__ = ["crop_transform", "safe_augmentations", "light_augmentations", "medium_augmentations", "hard_augmentations"]

from typing import Tuple


def crop_transform(image_size: Tuple[int, int], min_scale=0.75, max_scale=1.25, input_size=5000):
    return A.OneOrOther(
        A.RandomSizedCrop(
            (int(image_size[0] * min_scale), int(min(input_size, image_size[0] * max_scale))),
            image_size[0],
            image_size[1],
        ),
        A.CropNonEmptyMaskIfExists(image_size[0], image_size[1]),
    )


def crop_transform_xview2(image_size: Tuple[int, int], min_scale=0.4, max_scale=0.75, input_size=1024):
    return A.OneOrOther(
        A.RandomSizedCrop(
            (int(image_size[0] * min_scale), int(min(input_size, image_size[0] * max_scale))),
            image_size[0],
            image_size[1],
        ),
        A.Compose(
            [A.Resize(input_size * 2, input_size * 2), A.CropNonEmptyMaskIfExists(image_size[0], image_size[1])]
        ),
    )


def safe_augmentations():

    return A.Compose(
        [
            A.HorizontalFlip(),
            A.MaskDropout(5),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(),
                    A.CLAHE(),
                    A.HueSaturationValue(),
                    A.RGBShift(),
                    A.RandomGamma(),
                    A.NoOp(),
                ]
            ),
            A.Normalize(),
        ]
    )


def light_augmentations():
    return A.Compose(
        [
            A.HorizontalFlip(),
            A.MaskDropout(5),
            A.ShiftScaleRotate(scale_limit=0.05, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT),
            A.RandomBrightnessContrast(),
            A.HueSaturationValue(),
            A.CLAHE(),
            A.RGBShift(),
            A.RandomGamma(),
            A.Normalize(),
        ]
    )


def medium_augmentations():
    return A.Compose(
        [
            A.HorizontalFlip(),
            A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT),
            # Add occasion blur/sharpening
            A.OneOf([A.GaussianBlur(), A.IAASharpen(), A.NoOp()]),
            # Spatial-preserving augmentations:
            A.OneOf([A.CoarseDropout(), A.MaskDropout(max_objects=5), A.NoOp()]),
            A.GaussNoise(),
            A.OneOf([A.RandomBrightnessContrast(), A.CLAHE(), A.HueSaturationValue(), A.RGBShift(), A.RandomGamma()]),
            # Weather effects
            A.RandomFog(fog_coef_lower=0.01, fog_coef_upper=0.3, p=0.1),
            A.Normalize(),
        ]
    )


def hard_augmentations():
    return A.Compose(
        [
            A.RandomRotate90(),
            A.Transpose(),
            A.RandomGridShuffle(),
            A.ShiftScaleRotate(
                scale_limit=0.1, rotate_limit=45, border_mode=cv2.BORDER_CONSTANT, mask_value=0, value=0
            ),
            A.ElasticTransform(border_mode=cv2.BORDER_CONSTANT, alpha_affine=5, mask_value=0, value=0),
            # Add occasion blur
            A.OneOf([A.GaussianBlur(), A.GaussNoise(), A.IAAAdditiveGaussianNoise(), A.NoOp()]),
            # D4 Augmentations
            A.OneOf([A.CoarseDropout(), A.MaskDropout(max_objects=10), A.NoOp()]),
            # Spatial-preserving augmentations:
            A.OneOf(
                [
                    A.RandomBrightnessContrast(brightness_by_max=True),
                    A.CLAHE(),
                    A.HueSaturationValue(),
                    A.RGBShift(),
                    A.RandomGamma(),
                    A.NoOp(),
                ]
            ),
            # Weather effects
            A.OneOf([A.RandomFog(fog_coef_lower=0.01, fog_coef_upper=0.3, p=0.1), A.NoOp()]),
            A.Normalize(),
        ]
    )
