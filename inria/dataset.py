import math
import os
from typing import List, Callable, Optional

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from pytorch_toolbelt.inference.tiles import ImageSlicer
from pytorch_toolbelt.utils import fs
from pytorch_toolbelt.utils.catalyst import PseudolabelDatasetMixin
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image, tensor_from_mask_image
from scipy.ndimage import binary_dilation, binary_fill_holes
from torch.utils.data import WeightedRandomSampler, Dataset, ConcatDataset

from .augmentations import *

INPUT_IMAGE_KEY = "image"
INPUT_IMAGE_ID_KEY = "image_id"
INPUT_MASK_KEY = "mask"
OUTPUT_MASK_KEY = "mask"
INPUT_INDEX_KEY = "index"

# Smaller masks for deep supervision
OUTPUT_MASK_4_KEY = "mask_4"
OUTPUT_MASK_8_KEY = "mask_8"
OUTPUT_MASK_16_KEY = "mask_16"
OUTPUT_MASK_32_KEY = "mask_32"

OUTPUT_CLASS_KEY = "classes"

UNLABELED_SAMPLE = 127

# NOISY SAMPLES
# chicago27
# vienna30
# austin23
# chicago26

TRAIN_LOCATIONS = ["austin", "chicago", "kitsap", "tyrol-w", "vienna"]
TEST_LOCATIONS = ["bellingham", "bloomington", "innsbruck", "sfo", "tyrol-e"]


def read_inria_image(fname):
    image = cv2.imread(fname)
    return image


def read_inria_mask(fname):
    mask = fs.read_image_as_is(fname)
    cv2.threshold(mask, thresh=0, maxval=1, type=cv2.THRESH_BINARY, dst=mask)
    return mask


def read_inria_mask_with_pseudolabel(fname):
    mask = fs.read_image_as_is(fname)
    mask[mask > UNLABELED_SAMPLE] = 1
    return mask


def compute_boundary_mask(mask: np.ndarray) -> np.ndarray:
    dilated = binary_dilation(mask, structure=np.ones((5, 5), dtype=np.bool))
    dilated = binary_fill_holes(dilated)

    diff = dilated & ~mask
    diff = cv2.dilate(diff, kernel=(5, 5))
    diff = diff & ~mask
    return diff.astype(np.uint8)


class InriaImageMaskDataset(Dataset, PseudolabelDatasetMixin):
    def __init__(
        self,
        image_filenames: List[str],
        mask_filenames: Optional[List[str]],
        transform: A.Compose,
        image_loader=read_inria_image,
        mask_loader=read_inria_mask,
        use_edges=False,
    ):
        if mask_filenames is not None and len(image_filenames) != len(mask_filenames):
            raise ValueError("Number of images does not corresponds to number of targets")

        self.image_ids = [fs.id_from_fname(fname) for fname in image_filenames]
        self.use_edges = use_edges

        self.images = image_filenames
        self.masks = mask_filenames
        self.get_image = image_loader
        self.get_mask = mask_loader

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def set_target(self, index: int, value: np.ndarray):
        mask_fname = self.masks[index]

        value = value.astype(np.uint8)
        value[value == 1] = 255
        cv2.imwrite(mask_fname, value)

    def __getitem__(self, index):
        image = self.get_image(self.images[index])

        if self.masks is not None:
            mask = self.get_mask(self.masks[index])
        else:
            mask = np.ones((image.shape[0], image.shape[1], 1), dtype=np.uint8) * UNLABELED_SAMPLE

        data = self.transform(image=image, mask=mask)

        sample = {
            INPUT_IMAGE_KEY: tensor_from_rgb_image(data["image"]),
            INPUT_IMAGE_ID_KEY: self.image_ids[index],
            INPUT_INDEX_KEY: index,
            INPUT_MASK_KEY: tensor_from_mask_image(data["mask"]).float(),
        }

        return sample


class _InrialTiledImageMaskDataset(Dataset):
    def __init__(
        self,
        image_fname: str,
        mask_fname: str,
        image_loader: Callable,
        target_loader: Callable,
        tile_size,
        tile_step,
        image_margin=0,
        transform=None,
        target_shape=None,
        use_edges=False,
        keep_in_mem=False,
    ):
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
                raise ValueError(
                    f"Image size {image.shape} and mask shape {image.shape} must have equal width and height"
                )

            target_shape = image.shape

        self.slicer = ImageSlicer(target_shape, tile_size, tile_step, image_margin)

        self.transform = transform
        self.image_ids = [fs.id_from_fname(image_fname)] * len(self.slicer.crops)
        self.crop_coords_str = [f"[{crop[0]};{crop[1]};{crop[2]};{crop[3]};]" for crop in self.slicer.crops]

    def _get_image(self, index):
        image = self.image_loader(self.image_fname)
        image = self.slicer.cut_patch(image, index)
        return image

    def _get_mask(self, index):
        mask = self.mask_loader(self.mask_fname)
        mask = self.slicer.cut_patch(mask, index)
        return mask

    def __len__(self):
        return len(self.slicer.crops)

    def __getitem__(self, index):
        image = self._get_image(index)
        mask = self._get_mask(index)
        data = self.transform(image=image, mask=mask)

        image = data["image"]
        mask = data["mask"]

        # coarse_mask = cv2.resize(
        #     mask,
        #     dsize=(mask.shape[1] // 4, mask.shape[0] // 4),
        #     interpolation=cv2.INTER_LINEAR,
        # )

        data = {
            INPUT_IMAGE_KEY: tensor_from_rgb_image(image),
            INPUT_MASK_KEY: tensor_from_mask_image(mask).float(),
            # "coarse_targets": tensor_from_mask_image(coarse_mask).float(),
            INPUT_IMAGE_ID_KEY: self.image_ids[index],
            "crop_coords": self.crop_coords_str[index],
        }

        if self.use_edges:
            data["edge"] = tensor_from_mask_image(compute_boundary_mask(mask)).float()

        return data


class InrialTiledImageMaskDataset(ConcatDataset):
    def __init__(
        self,
        image_filenames: List[str],
        target_filenames: List[str],
        image_loader=read_inria_image,
        target_loader=read_inria_mask,
        use_edges=False,
        **kwargs,
    ):
        if len(image_filenames) != len(target_filenames):
            raise ValueError("Number of images does not corresponds to number of targets")

        datasets = []
        for image, mask in zip(image_filenames, target_filenames):
            dataset = _InrialTiledImageMaskDataset(
                image, mask, image_loader, target_loader, use_edges=use_edges, **kwargs
            )
            datasets.append(dataset)
        super().__init__(datasets)


def get_datasets(
    data_dir: str,
    image_size=(224, 224),
    augmentation="hard",
    train_mode="random",
    use_edges=False,
    sanity_check=False,
    fast=False,
):
    """
    Create train and validation data loaders
    :param data_dir: Inria dataset directory
    :param fast: Fast training model. Use only one image per location for training and one image per location for validation
    :param image_size: Size of image crops during training & validation
    :param use_edges: If True, adds 'edge' target mask
    :param augmentation: Type of image augmentations to use
    :param train_mode:
    'random' - crops tiles from source images randomly.
    'tiles' - crop image in overlapping tiles (guaranteed to process entire dataset)
    :return: (train_loader, valid_loader)
    """

    if augmentation == "hard":
        train_transform = hard_augmentations()
    elif augmentation == "medium":
        train_transform = medium_augmentations()
    elif augmentation == "light":
        train_transform = light_augmentations()
    elif augmentation == "safe":
        train_transform = safe_augmentations()
    else:
        train_transform = A.Normalize()

    valid_transform = A.Normalize()
    assert train_mode in {"random", "tiles"}
    if train_mode == "random":
        locations = TRAIN_LOCATIONS

        # Empirical weights for regions (for oversampling hard regions)
        # locations_weights = [1, 2, 3, 1, 1]
        locations_weights = [1, 1, 1, 1, 1]

        train_data = []
        train_data_weights = []
        valid_data = []

        # For validation, we remove the first five images of every location (e.g., austin{1-5}.tif, chicago{1-5}.tif) from the training set.
        # That is suggested validation strategy by competition host

        if fast:
            # Fast training model. Use only one image per location for training and one image per location for validation
            for loc, weight in zip(locations, locations_weights):
                valid_data.append(f"{loc}1")
                train_data.append(f"{loc}6")
                train_data_weights.append(weight)
        else:
            for loc, weight in zip(locations, locations_weights):
                for i in range(1, 6):
                    valid_data.append(f"{loc}{i}")
                for i in range(6, 37):
                    train_data.append(f"{loc}{i}")
                    train_data_weights.append(weight)

        train_img = [os.path.join(data_dir, "train", "images", f"{fname}.tif") for fname in train_data]
        valid_img = [os.path.join(data_dir, "train", "images", f"{fname}.tif") for fname in valid_data]

        train_mask = [os.path.join(data_dir, "train", "gt", f"{fname}.tif") for fname in train_data]
        valid_mask = [os.path.join(data_dir, "train", "gt", f"{fname}.tif") for fname in valid_data]

        train_transform = A.Compose([crop_transform(image_size), train_transform])

        trainset = InriaImageMaskDataset(train_img, train_mask, use_edges=use_edges, transform=train_transform)
        num_train_samples = int(len(trainset) * (5000 * 5000) / (image_size[0] * image_size[1]))
        train_sampler = WeightedRandomSampler(np.array(train_data_weights), num_train_samples)

        validset = InrialTiledImageMaskDataset(
            valid_img,
            valid_mask,
            use_edges=use_edges,
            transform=valid_transform,
            # For validation we don't want tiles overlap
            tile_size=image_size,
            tile_step=image_size,
            target_shape=(5000, 5000),
        )

    elif train_mode == "tiles":
        inria_tiles = pd.read_csv(os.path.join(data_dir, "inria_tiles.csv"))

        train_img = inria_tiles[inria_tiles["train"] == 1]["image"].tolist()
        train_mask = inria_tiles[inria_tiles["train"] == 1]["mask"].tolist()

        valid_img = inria_tiles[inria_tiles["train"] == 0]["image"].tolist()
        valid_mask = inria_tiles[inria_tiles["train"] == 0]["mask"].tolist()

        train_img = [os.path.join(data_dir, x) for x in train_img]
        train_mask = [os.path.join(data_dir, x) for x in train_mask]

        valid_img = [os.path.join(data_dir, x) for x in valid_img]
        valid_mask = [os.path.join(data_dir, x) for x in valid_mask]

        if fast:
            train_img = train_img[:128]
            train_mask = train_mask[:128]

            valid_img = valid_img[:128]
            valid_mask = valid_mask[:128]

        trainset = InriaImageMaskDataset(train_img, train_mask, use_edges=use_edges, transform=train_transform)
        validset = InriaImageMaskDataset(valid_img, valid_mask, use_edges=use_edges, transform=valid_transform)
        train_sampler = None
    else:
        raise ValueError(train_mode)

    if sanity_check:
        first_batch = [trainset[i] for i in range(32)]
        return first_batch * 50, first_batch, None

    return trainset, validset, train_sampler


def get_pseudolabeling_dataset(data_dir: str, include_masks: bool, image_size=(224, 224), augmentation=None):
    images = fs.find_images_in_dir(os.path.join(data_dir, "test_tiles", "images"))

    masks_dir = os.path.join(data_dir, "test_tiles", "masks")
    os.makedirs(masks_dir, exist_ok=True)

    masks = [os.path.join(masks_dir, fs.id_from_fname(image_fname) + ".png") for image_fname in images]

    is_whole_image_input = False

    if augmentation == "hard":
        transfrom = hard_augmentations()
    elif augmentation == "medium":
        transfrom = medium_augmentations()
    elif augmentation == "light":
        transfrom = light_augmentations()
    else:
        transfrom = A.Normalize()

    return InriaImageMaskDataset(
        images,
        masks if include_masks else None,
        transform=transfrom,
        image_loader=read_inria_image,
        mask_loader=read_inria_mask_with_pseudolabel,
    )
