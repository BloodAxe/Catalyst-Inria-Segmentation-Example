import os
from typing import List, Callable

import math
import pandas as pd
import albumentations as A
import cv2
import numpy as np
from pytorch_toolbelt.inference.tiles import ImageSlicer
from pytorch_toolbelt.utils import fs
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image, tensor_from_mask_image
from scipy.ndimage import binary_dilation, binary_fill_holes
from torch.utils.data import WeightedRandomSampler, DataLoader, Dataset, ConcatDataset


def read_inria_rgb(fname):
    image = cv2.imread(fname)
    cv2.cvtColor(image, code=cv2.COLOR_BGR2RGB, dst=image)
    return image


def read_inria_mask(fname):
    mask = fs.read_image_as_is(fname)
    cv2.threshold(mask, thresh=0, maxval=1, type=cv2.THRESH_BINARY, dst=mask)
    return mask


def padding_for_rotation(image_size, rotation):
    r = math.sqrt((image_size[0] / 2) ** 2 + (image_size[1] / 2) ** 2)

    rot_angle_rads = math.radians(45 - rotation)

    pad_h = int(r * math.cos(rot_angle_rads) - image_size[0] // 2)
    pad_w = int(r * math.cos(rot_angle_rads) - image_size[1] // 2)

    print('Image padding for rotation', rotation, pad_w, pad_h, r)
    return pad_h, pad_w


def compute_boundary_mask(mask: np.ndarray) -> np.ndarray:
    dilated = binary_dilation(mask, structure=np.ones((5, 5), dtype=np.bool))
    dilated = binary_fill_holes(dilated)

    diff = dilated & ~mask
    diff = cv2.dilate(diff, kernel=(5, 5))
    diff = diff & ~mask
    return diff.astype(np.uint8)


class InriaImageMaskDataset(Dataset):
    def __init__(self, image_filenames, target_filenames, image_loader, target_loader, transform=None, keep_in_mem=False, use_edges=False):
        if len(image_filenames) != len(target_filenames):
            raise ValueError('Number of images does not corresponds to number of targets')

        self.image_ids = [fs.id_from_fname(fname) for fname in image_filenames]
        self.use_edges = use_edges

        if keep_in_mem:
            self.images = [image_loader(fname) for fname in image_filenames]
            self.masks = [target_loader(fname) for fname in target_filenames]
            self.get_image = lambda x: x
            self.get_loader = lambda x: x
        else:
            self.images = image_filenames
            self.masks = target_filenames
            self.get_image = image_loader
            self.get_loader = target_loader

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.get_image(self.images[index])
        mask = self.get_loader(self.masks[index])

        data = self.transform(image=image, mask=mask)

        image = data['image']
        mask = data['mask']

        coarse_mask = cv2.resize(mask,
                                 dsize=(mask.shape[1] // 4, mask.shape[0] // 4),
                                 interpolation=cv2.INTER_LINEAR)

        data = {'features': tensor_from_rgb_image(image),
                'targets': tensor_from_mask_image(mask).float(),
                'coarse_targets': tensor_from_mask_image(coarse_mask).float(),
                'image_id': self.image_ids[index]}

        if self.use_edges:
            data['edge'] = tensor_from_mask_image(compute_boundary_mask(mask)).float()

        return data


class _InrialTiledImageMaskDataset(Dataset):
    def __init__(self, image_fname: str,
                 mask_fname: str,
                 image_loader: Callable,
                 target_loader: Callable,
                 tile_size,
                 tile_step,
                 image_margin=0,
                 transform=None,
                 target_shape=None,
                 use_edges=False,
                 keep_in_mem=False):
        self.image_fname = image_fname
        self.mask_fname = mask_fname
        self.image_loader = image_loader
        self.mask_loader = target_loader
        self.image = None
        self.mask = None
        self.use_edges = use_edges

        if target_shape is None or keep_in_mem:
            image = image_loader(image_fname)
            mask = target_loader(mask_fname)
            if image.shape[0] != mask.shape[0] or image.shape[1] != mask.shape[1]:
                raise ValueError(f"Image size {image.shape} and mask shape {image.shape} must have equal width and height")

            target_shape = image.shape

        self.slicer = ImageSlicer(target_shape, tile_size, tile_step, image_margin)

        if keep_in_mem:
            self.images = self.slicer.split(image)
            self.masks = self.slicer.split(mask)
        else:
            self.images = None
            self.masks = None

        self.transform = transform
        self.image_ids = [fs.id_from_fname(image_fname)] * len(self.slicer.crops)
        self.crop_coords_str = [f'[{crop[0]};{crop[1]};{crop[2]};{crop[3]};]' for crop in self.slicer.crops]

    def _get_image(self, index):
        if self.images is None:
            image = self.image_loader(self.image_fname)
            image = self.slicer.cut_patch(image, index)
        else:
            image = self.images[index]
        return image

    def _get_mask(self, index):
        if self.masks is None:
            mask = self.mask_loader(self.mask_fname)
            mask = self.slicer.cut_patch(mask, index)
        else:
            mask = self.masks[index]
        return mask

    def __len__(self):
        return len(self.slicer.crops)

    def __getitem__(self, index):
        image = self._get_image(index)
        mask = self._get_mask(index)
        data = self.transform(image=image, mask=mask)

        image = data['image']
        mask = data['mask']
        coarse_mask = cv2.resize(mask,
                                 dsize=(mask.shape[1] // 4, mask.shape[0] // 4),
                                 interpolation=cv2.INTER_LINEAR)

        data = {'features': tensor_from_rgb_image(image),
                'targets': tensor_from_mask_image(mask).float(),
                'coarse_targets': tensor_from_mask_image(coarse_mask).float(),
                'image_id': self.image_ids[index],
                'crop_coords': self.crop_coords_str[index]}

        if self.use_edges:
            data['edge'] = tensor_from_mask_image(compute_boundary_mask(mask)).float()

        return data


class InrialTiledImageMaskDataset(ConcatDataset):
    def __init__(self,
                 image_filenames: List[str],
                 target_filenames: List[str],
                 image_loader: Callable,
                 target_loader: Callable,
                 use_edges=False,
                 **kwargs):
        if len(image_filenames) != len(target_filenames):
            raise ValueError('Number of images does not corresponds to number of targets')

        datasets = []
        for image, mask in zip(image_filenames, target_filenames):
            dataset = _InrialTiledImageMaskDataset(image, mask, image_loader, target_loader, use_edges=use_edges, **kwargs)
            datasets.append(dataset)
        super().__init__(datasets)


def light_augmentations(image_size, whole_image_input=True, rot_angle=5):
    if whole_image_input:

        pad_h, pad_w = padding_for_rotation(image_size, rot_angle)
        crop_height = int(image_size[0] + pad_h * 2)
        crop_width = int(image_size[1] + pad_w * 2)

        spatial_transform = A.Compose([
            # Make random-sized crop with scale [75%..125%] of target size 1.5 larger than target crop to have some space around for
            # further transforms
            A.RandomSizedCrop((int(crop_height * 0.75), int(crop_height * 1.25)), crop_height, crop_width),

            # Apply random rotations
            A.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=rot_angle, border_mode=cv2.BORDER_CONSTANT),

            # Crop to desired image size
            A.CenterCrop(image_size[0], image_size[1]),
        ])
    else:
        spatial_transform = A.Compose([
            A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT),
        ])

    return A.Compose([
        spatial_transform,

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

        A.Normalize()
    ])


def medium_augmentations(image_size, whole_image_input=True, rot_angle=15):
    if whole_image_input:

        pad_h, pad_w = padding_for_rotation(image_size, rot_angle)
        crop_height = int(image_size[0] + pad_h * 2)
        crop_width = int(image_size[1] + pad_w * 2)

        spatial_transform = A.Compose([
            # Make random-sized crop with scale [75%..125%] of target size 1.5 larger than target crop to have some space around for
            # further transforms
            A.RandomSizedCrop((int(crop_height * 0.75), int(crop_height * 1.25)), crop_height, crop_width),

            # Apply random rotations
            A.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=rot_angle, border_mode=cv2.BORDER_CONSTANT),

            # Crop to desired image size
            A.CenterCrop(image_size[0], image_size[1]),
        ])
    else:
        spatial_transform = A.Compose([
            A.ShiftScaleRotate(scale_limit=0.25, rotate_limit=rot_angle, border_mode=cv2.BORDER_CONSTANT),
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


def hard_augmentations(image_size, whole_image_input=True, rot_angle=45):
    if whole_image_input:
        pad_h, pad_w = padding_for_rotation(image_size, rot_angle)
        crop_height = int(image_size[0] + pad_h * 2)
        crop_width = int(image_size[1] + pad_w * 2)

        spatial_transform = A.Compose([
            # Make random-sized crop with scale [50%..200%] of crop size to have some space around for
            # further transforms
            A.RandomSizedCrop((int(crop_height * 0.5), int(crop_height * 2)), crop_height, crop_width),

            # Apply random rotations
            A.OneOf([
                A.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=rot_angle, border_mode=cv2.BORDER_CONSTANT),
                A.IAAAffine(shear=0.1, rotate=rot_angle, mode='constant'),
            ]),

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
                    use_edges=False,
                    sanity_check=False,
                    fast=False):
    """
    Create train and validation data loaders
    :param data_dir: Inria dataset directory
    :param batch_size:
    :param num_workers:
    :param fast: Fast training model. Use only one image per location for training and one image per location for validation
    :param image_size: Size of image crops during training & validation
    :param use_edges: If True, adds 'edge' target mask
    :param augmentation: Type of image augmentations to use
    :param train_mode:
    'random' - crops tiles from source images randomly.
    'tiles' - crop image in overlapping tiles (guaranteed to process entire dataset)
    :return: (train_loader, valid_loader)
    """

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

    valid_transform = A.Normalize()

    if train_mode == 'random':
        locations = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']

        train_data = []
        valid_data = []

        # For validation, we remove the first five images of every location (e.g., austin{1-5}.tif, chicago{1-5}.tif) from the training set.
        # That is suggested validation strategy by competition host

        if fast:
            # Fast training model. Use only one image per location for training and one image per location for validation
            for loc in locations:
                valid_data.append(f'{loc}1')
                train_data.append(f'{loc}6')
        else:
            for loc in locations:
                for i in range(1, 6):
                    valid_data.append(f'{loc}{i}')
                for i in range(6, 37):
                    train_data.append(f'{loc}{i}')

        train_img = [os.path.join(data_dir, 'train', 'images', f'{fname}.tif') for fname in train_data]
        valid_img = [os.path.join(data_dir, 'train', 'images', f'{fname}.tif') for fname in valid_data]

        train_mask = [os.path.join(data_dir, 'train', 'gt', f'{fname}.tif') for fname in train_data]
        valid_mask = [os.path.join(data_dir, 'train', 'gt', f'{fname}.tif') for fname in valid_data]

        trainset = InriaImageMaskDataset(train_img, train_mask, read_inria_rgb, read_inria_mask,
                                         use_edges=use_edges,
                                         transform=train_transform,
                                         keep_in_mem=False)
        num_train_samples = int(len(trainset) * (5000 * 5000) / (image_size[0] * image_size[1]))
        train_sampler = WeightedRandomSampler(np.ones(len(trainset)), num_train_samples)

        validset = InrialTiledImageMaskDataset(valid_img, valid_mask,
                                               use_edges=use_edges,
                                               image_loader=read_inria_rgb,
                                               target_loader=read_inria_mask,
                                               transform=valid_transform,
                                               # For validation we don't want tiles overlap
                                               tile_size=image_size,
                                               tile_step=image_size,
                                               target_shape=(5000, 5000),
                                               keep_in_mem=False)

    elif train_mode == 'tiles':
        inria_tiles = pd.read_csv('inria_tiles.csv')

        train_img = inria_tiles[inria_tiles['train'] == 1]['image']
        train_mask = inria_tiles[inria_tiles['train'] == 1]['mask']

        valid_img = inria_tiles[inria_tiles['train'] == 0]['image']
        valid_mask = inria_tiles[inria_tiles['train'] == 0]['mask']

        trainset = InriaImageMaskDataset(train_img, train_mask,
                                         use_edges=use_edges,
                                         image_loader=read_inria_rgb,
                                         target_loader=read_inria_mask,
                                         transform=train_transform,
                                         keep_in_mem=False)

        validset = InriaImageMaskDataset(valid_img, valid_mask,
                                         use_edges=use_edges,
                                         image_loader=read_inria_rgb,
                                         target_loader=read_inria_mask,
                                         transform=valid_transform,
                                         keep_in_mem=False)
        train_sampler = None
    else:
        raise ValueError(train_mode)

    if sanity_check:
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
