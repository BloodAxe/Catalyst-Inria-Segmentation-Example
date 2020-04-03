from multiprocessing.pool import Pool
from typing import List, Dict

import albumentations as A
import cv2
import numpy as np
import torch
from pytorch_toolbelt.inference.tiles import CudaTileMerger, ImageSlicer
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image, to_numpy
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from inria.dataset import OUTPUT_MASK_KEY


class InMemoryDataset(Dataset):
    def __init__(self, data: List[Dict], transform: A.Compose):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.transform(**self.data[item])


def _tensor_from_rgb_image(image: np.ndarray, **kwargs):
    return tensor_from_rgb_image(image)


class PickModelOutput(nn.Module):
    def __init__(self, model, key):
        super().__init__()
        self.model = model
        self.target_key = key

    def forward(self, input):
        output = self.model(input)
        return output[self.target_key]


@torch.no_grad()
def predict(model: nn.Module, image: np.ndarray, image_size, normalize=A.Normalize(), batch_size=1) -> np.ndarray:

    tile_step = (image_size[0] // 2, image_size[1] // 2)

    tile_slicer = ImageSlicer(image.shape, image_size, tile_step)
    tile_merger = CudaTileMerger(tile_slicer.target_shape, 1, tile_slicer.weight)
    patches = tile_slicer.split(image)

    transform = A.Compose([normalize, A.Lambda(image=_tensor_from_rgb_image)])

    data = list(
        {"image": patch, "coords": np.array(coords, dtype=np.int)}
        for (patch, coords) in zip(patches, tile_slicer.crops)
    )
    for batch in DataLoader(InMemoryDataset(data, transform), pin_memory=True, batch_size=batch_size):
        image = batch["image"].cuda(non_blocking=True)
        coords = batch["coords"]
        mask_batch = model(image)
        tile_merger.integrate_batch(mask_batch[OUTPUT_MASK_KEY], coords)

    mask = tile_merger.merge()

    mask = np.moveaxis(to_numpy(mask), 0, -1)
    mask = tile_slicer.crop_to_orignal_size(mask)

    return mask


def __compute_ious(args):
    thresholds = np.arange(0, 256)
    gt, pred = args
    gt = cv2.imread(gt) > 0  # Make binary {0,1}
    pred = cv2.imread(pred)

    pred_i = np.zeros_like(gt)

    intersection = np.zeros(len(thresholds))
    union = np.zeros(len(thresholds))

    gt_sum = gt.sum()
    for index, threshold in enumerate(thresholds):
        np.greater(pred, threshold, out=pred_i)
        union[index] += gt_sum + pred_i.sum()

        np.logical_and(gt, pred_i, out=pred_i)
        intersection[index] += pred_i.sum()

    return intersection, union


def optimize_threshold(gt_images, pred_images):
    thresholds = np.arange(0, 256)

    intersection = np.zeros(len(thresholds))
    union = np.zeros(len(thresholds))

    with Pool(32) as wp:
        for i, u in tqdm(wp.imap_unordered(__compute_ious, zip(gt_images, pred_images)), total=len(gt_images)):
            intersection += i
            union += u

    return thresholds, intersection / (union - intersection)
