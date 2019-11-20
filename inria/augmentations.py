import math

import albumentations as A
import cv2

__all__ = ["light_augmentations", "medium_augmentations", "hard_augmentations"]


def padding_for_rotation(image_size, rotation):
    r = math.sqrt((image_size[0] / 2) ** 2 + (image_size[1] / 2) ** 2)

    rot_angle_rads = math.radians(45 - rotation)

    pad_h = int(r * math.cos(rot_angle_rads) - image_size[0] // 2)
    pad_w = int(r * math.cos(rot_angle_rads) - image_size[1] // 2)

    print("Image padding for rotation", rotation, pad_w, pad_h, r)
    return pad_h, pad_w


def light_augmentations(image_size, whole_image_input=True, rot_angle=5):
    if whole_image_input:

        pad_h, pad_w = padding_for_rotation(image_size, rot_angle)
        crop_height = int(image_size[0] + pad_h * 2)
        crop_width = int(image_size[1] + pad_w * 2)

        spatial_transform = A.Compose(
            [
                # Make random-sized crop with scale [75%..125%] of target size 1.5 larger than target crop to have some space around for
                # further transforms
                A.RandomSizedCrop((int(crop_height * 0.75), int(crop_height * 1.25)), crop_height, crop_width),
                # Apply random rotations
                A.ShiftScaleRotate(
                    shift_limit=0, scale_limit=0, rotate_limit=rot_angle, border_mode=cv2.BORDER_CONSTANT
                ),
                # Crop to desired image size
                A.CenterCrop(image_size[0], image_size[1]),
            ]
        )
    else:
        spatial_transform = A.Compose(
            [A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT)]
        )

    return A.Compose(
        [
            spatial_transform,
            # D4 Augmentations
            A.Compose([A.Transpose(), A.RandomRotate90()]),
            # Spatial-preserving augmentations:
            A.OneOf([A.CoarseDropout(), A.GaussNoise()]),
            A.OneOf([A.RandomBrightnessContrast(), A.CLAHE(), A.HueSaturationValue(), A.RGBShift(), A.RandomGamma()]),
            A.Normalize(),
        ]
    )


def medium_augmentations(image_size, whole_image_input=True, rot_angle=15):
    if whole_image_input:

        pad_h, pad_w = padding_for_rotation(image_size, rot_angle)
        crop_height = int(image_size[0] + pad_h * 2)
        crop_width = int(image_size[1] + pad_w * 2)

        spatial_transform = A.Compose(
            [
                # Make random-sized crop with scale [75%..125%] of target size 1.5 larger than target crop to have some space around for
                # further transforms
                A.RandomSizedCrop((int(crop_height * 0.75), int(crop_height * 1.25)), crop_height, crop_width),
                # Apply random rotations
                A.ShiftScaleRotate(
                    shift_limit=0, scale_limit=0, rotate_limit=rot_angle, border_mode=cv2.BORDER_CONSTANT
                ),
                # Crop to desired image size
                A.CenterCrop(image_size[0], image_size[1]),
            ]
        )
    else:
        spatial_transform = A.Compose(
            [A.ShiftScaleRotate(scale_limit=0.25, rotate_limit=rot_angle, border_mode=cv2.BORDER_CONSTANT)]
        )

    return A.Compose(
        [
            spatial_transform,
            # Add occasion blur/sharpening
            A.OneOf([A.GaussianBlur(), A.MotionBlur(), A.IAASharpen()]),
            # D4 Augmentations
            A.Compose([A.Transpose(), A.RandomRotate90()]),
            # Spatial-preserving augmentations:
            A.OneOf([A.CoarseDropout(), A.MaskDropout(max_objects=3), A.NoOp()]),
            A.GaussNoise(),
            A.OneOf([A.RandomBrightnessContrast(), A.CLAHE(), A.HueSaturationValue(), A.RGBShift(), A.RandomGamma()]),
            # Weather effects
            A.OneOf([A.RandomFog(fog_coef_lower=0.01, fog_coef_upper=0.3, p=0.1), A.NoOp()]),
            A.Normalize(),
        ]
    )


def hard_augmentations(image_size, whole_image_input=True, rot_angle=45):
    if whole_image_input:
        pad_h, pad_w = padding_for_rotation(image_size, rot_angle)
        crop_height = int(image_size[0] + pad_h * 2)
        crop_width = int(image_size[1] + pad_w * 2)

        spatial_transform = A.Compose(
            [
                # Make random-sized crop with scale [50%..200%] of crop size to have some space around for
                # further transforms
                A.RandomSizedCrop((int(crop_height * 0.5), int(crop_height * 2)), crop_height, crop_width),
                # Apply random rotations
                A.OneOf(
                    [
                        A.ShiftScaleRotate(
                            shift_limit=0, scale_limit=0, rotate_limit=rot_angle, border_mode=cv2.BORDER_CONSTANT
                        ),
                        A.IAAAffine(shear=0.1, rotate=rot_angle, mode="constant"),
                        A.NoOp(),
                    ]
                ),
                # Crop to desired image size
                A.CenterCrop(image_size[0], image_size[1]),
            ]
        )
    else:
        spatial_transform = A.Compose(
            [A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=45, border_mode=cv2.BORDER_CONSTANT), A.NoOp()]
        )

    return A.Compose(
        [
            spatial_transform,
            # Add occasion blur
            A.OneOf([A.GaussianBlur(), A.GaussNoise(), A.IAAAdditiveGaussianNoise(), A.NoOp()]),
            # D4 Augmentations
            A.Compose([A.Transpose(), A.RandomRotate90()]),
            A.OneOf([A.CoarseDropout(), A.MaskDropout(max_objects=3), A.NoOp()]),
            # Spatial-preserving augmentations:
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
            # Weather effects
            A.OneOf([A.RandomFog(fog_coef_lower=0.01, fog_coef_upper=0.3, p=0.1), A.NoOp()]),
            A.Normalize(),
        ]
    )
