import os

import albumentations as A
import cv2
import numpy as np
from pytorch_toolbelt.utils.dataset_utils import TiledImageMaskDataset, ImageMaskDataset
from pytorch_toolbelt.utils.fs import read_rgb_image, read_image_as_is
from torch.utils.data import WeightedRandomSampler, DataLoader


def read_inria_mask(fname):
    mask = read_image_as_is(fname)
    return (mask > 0).astype(np.uint8)


def light_augmentations(image_size, whole_image_input=True):
    return A.Compose([
        # Make random-sized crop with scale [50%..200%] of target size 1.5 larger than target crop to have some space around for
        # further transforms
        A.RandomSizedCrop((int(image_size[0] * 0.75), int(image_size[1] * 1.25)), image_size[0], image_size[1]),

        # Apply random rotations
        A.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=5, border_mode=cv2.BORDER_CONSTANT),

        # Crop to desired image size
        A.CenterCrop(image_size[0], image_size[1]),

        # Spatial-preserving augmentations:
        A.OneOf([
            A.Cutout(),
            A.GaussNoise(),
        ]),

        A.OneOf([
            A.RandomBrightnessContrast(),
            A.CLAHE(),
            A.HueSaturationValue(),
            A.RGBShift(),
            A.RandomGamma()
        ]),

        A.Normalize()
    ])


def medium_augmentations(image_size, whole_image_input=True):
    return A.Compose([
        # Make random-sized crop with scale [75%..125%] of target size 1.5 larger than target crop to have some space around for
        # further transforms
        A.RandomSizedCrop((int(image_size[0] * 0.75), int(image_size[1] * 1.25)), image_size[0], image_size[1]),

        # Apply random rotations
        A.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT),

        # Add occasion blur/sharpening
        A.OneOf([
            A.GaussianBlur(),
            A.MotionBlur(),
            A.IAASharpen()
        ]),

        # D4 Augmentations
        A.Compose([
            A.Transpose(),
            A.RandomRotate90(),
        ]),

        # Spatial-preserving augmentations:
        A.OneOf([
            A.Cutout(),
            A.GaussNoise(),
        ]),
        A.OneOf([
            A.RandomBrightnessContrast(),
            A.CLAHE(),
            A.HueSaturationValue(),
            A.RGBShift(),
            A.RandomGamma()
        ]),
        # Weather effects
        A.OneOf([
            A.RandomFog(fog_coef_lower=0.01, fog_coef_upper=0.3, p=0.1),
            A.NoOp()
        ]),

        A.Normalize()
    ])


def hard_augmentations(image_size, whole_image_input=True):
    if whole_image_input:
        spatial_transform = A.Compose([
            # Make random-sized crop with scale [50%..200%] of target size 1.5 larger than target crop to have some space around for
            # further transforms
            A.RandomSizedCrop((image_size[0] // 2, image_size[1] * 2), int(image_size[0] * 1.5), int(image_size[1] * 1.5)),

            # Apply random rotations
            A.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=45, border_mode=cv2.BORDER_CONSTANT),
            A.OneOf([
                A.GridDistortion(border_mode=cv2.BORDER_CONSTANT),
                A.ElasticTransform(alpha_affine=0, border_mode=cv2.BORDER_CONSTANT),
            ]),
            # Crop to desired image size
            A.CenterCrop(image_size[0], image_size[1]),
        ])
    else:
        spatial_transform = A.Compose([
            A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=45, border_mode=cv2.BORDER_CONSTANT),
            A.OneOf([
                A.GridDistortion(border_mode=cv2.BORDER_CONSTANT),
                A.ElasticTransform(alpha_affine=0, border_mode=cv2.BORDER_CONSTANT),
                A.NoOp()
            ])
        ])

    return A.Compose([
        spatial_transform,

        # Add occasion blur/sharpening
        A.OneOf([
            A.GaussianBlur(),
            A.MotionBlur(),
            A.IAASharpen()
        ]),

        # D4 Augmentations
        A.Compose([
            A.Transpose(),
            A.RandomRotate90(),
        ]),

        # Spatial-preserving augmentations:
        A.OneOf([
            A.Cutout(),
            A.GaussNoise(),
        ]),
        A.OneOf([
            A.RandomBrightnessContrast(),
            A.CLAHE(),
            A.HueSaturationValue(),
            A.RGBShift(),
            A.RandomGamma()
        ]),
        # Weather effects
        A.OneOf([
            A.RandomFog(fog_coef_lower=0.01, fog_coef_upper=0.3, p=0.1),
            A.NoOp()
        ]),

        A.Normalize()
    ])


def get_dataloaders(data_dir: str,
                    batch_size=16,
                    num_workers=4,
                    image_size=(224, 224),
                    augmentation='hard',
                    train_mode='random',
                    fast=False):
    """
    Create train and validation data loaders
    :param data_dir: Inria dataset directory
    :param batch_size:
    :param num_workers:
    :param fast: Fast training model. Use only one image per location for training and one image per location for validation
    :param image_size: Size of image crops during training & validation
    :param use_d4: Allows use of D4 augmentations for training; if False - will use horisontal flips (D4 may hurt the performance for off-nadir images)
    :param augmentation: Type of image augmentations to use
    :param train_mode: 'random' - crops tiles from source images randomly. 'tiles' - crop image in overlapping tiles (guaranteed to process entire dataset)
    :return: (train_loader, valid_loader)
    """
    locations = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']

    train_data = []
    valid_data = []

    # For validation, we remove the first five images of every location (e.g., austin{1-5}.tif, chicago{1-5}.tif) from the training set.
    # That is suggested validation strategy by competition host
    for loc in locations:
        for i in range(1, 6):
            valid_data.append(f'{loc}{i}')
        for i in range(6, 37):
            train_data.append(f'{loc}{i}')

    train_img = [os.path.join(data_dir, 'train', 'images', f'{fname}.tif') for fname in train_data]
    valid_img = [os.path.join(data_dir, 'train', 'images', f'{fname}.tif') for fname in valid_data]

    train_mask = [os.path.join(data_dir, 'train', 'gt', f'{fname}.tif') for fname in train_data]
    valid_mask = [os.path.join(data_dir, 'train', 'gt', f'{fname}.tif') for fname in valid_data]

    is_whole_image_input = train_mode == 'random'

    if augmentation == 'hard':
        train_transform = hard_augmentations(image_size, is_whole_image_input)
    elif augmentation == 'medium':
        train_transform = medium_augmentations(image_size, is_whole_image_input)
    elif augmentation == 'light':
        train_transform = light_augmentations(image_size, is_whole_image_input)
    else:
        assert not is_whole_image_input
        train_transform = A.Normalize()

    if train_mode == 'random':
        trainset = ImageMaskDataset(train_img, train_mask, read_rgb_image, read_inria_mask,
                                    transform=train_transform,
                                    keep_in_mem=False)
        num_train_samples = int(len(trainset) * (5000 * 5000) / (image_size[0] * image_size[1]))
        train_sampler = WeightedRandomSampler(np.ones(len(trainset)), num_train_samples)
    elif train_mode == 'all':
        trainset = TiledImageMaskDataset(train_img, train_mask, read_rgb_image, read_inria_mask,
                                         transform=train_transform,
                                         tile_size=image_size,
                                         tile_step=(image_size[0] // 2, image_size[1] // 2),
                                         target_shape=(5000, 5000),
                                         keep_in_mem=False)
        train_sampler = None
    else:
        raise ValueError(train_mode)

    validset = TiledImageMaskDataset(valid_img, valid_mask, read_rgb_image, read_inria_mask,
                                     transform=A.Normalize(),
                                     # For validation we don't want tiles overlap
                                     tile_size=image_size,
                                     tile_step=image_size,
                                     target_shape=(5000, 5000),
                                     keep_in_mem=False)

    if fast:
        first_batch = [trainset[i] for i in range(batch_size)]
        trainloader = DataLoader(first_batch * 50,
                                 batch_size=batch_size,
                                 pin_memory=True)
        validloader = DataLoader(first_batch,
                                 batch_size=batch_size,
                                 pin_memory=True)
        return trainloader, validloader

    trainloader = DataLoader(trainset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True,
                             shuffle=train_sampler is None,
                             sampler=train_sampler)

    validloader = DataLoader(validset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_memory=True,
                             shuffle=False)

    return trainloader, validloader
