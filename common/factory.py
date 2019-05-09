from functools import partial
from multiprocessing.pool import Pool
from typing import List, Dict

import albumentations as A
import cv2
import numpy as np
import torch
from pytorch_toolbelt.inference.tiles import CudaTileMerger, ImageSlicer
from pytorch_toolbelt import losses as L
from pytorch_toolbelt.inference.tta import TTAWrapper, fliplr_image2mask, d4_image2mask
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image, to_numpy, rgb_image_from_tensor
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from common.ternausnet2 import TernausNetV2
from .linknet import LinkNet152, LinkNet34
from .models import fpn_v2, fpn_v1, fpn_v3
from .unet import UNet
from pytorch_toolbelt.modules import encoders as E


def get_model(model_name: str, image_size=None) -> nn.Module:
    registry = {
        'unet': partial(UNet, upsample=False),
        'linknet34': LinkNet34,
        'linknet152': LinkNet152,
        'fpn128_mobilenetv3': partial(fpn_v2, encoder=E.MobilenetV3Encoder, fpn_features=128),
        'fpn128_resnet34': partial(fpn_v1, encoder=E.Resnet34Encoder, fpn_features=128),
        'fpn128_resnext50': partial(fpn_v1, encoder=E.SEResNeXt50Encoder, fpn_features=128),
        'fpn256_resnext50': partial(fpn_v1, encoder=E.SEResNeXt50Encoder, fpn_features=256),
        'fpn128_resnext50_v2': partial(fpn_v2, encoder=E.SEResNeXt50Encoder, fpn_features=128),
        'fpn256_resnext50_v2': partial(fpn_v2, encoder=E.SEResNeXt50Encoder, fpn_features=256),
        'fpn256_resnext50_v3': partial(fpn_v3, encoder=E.SEResNeXt50Encoder, fpn_features=256),
        'fpn256_senet154_v2': partial(fpn_v2, encoder=E.SENet154Encoder, fpn_features=256),

        'ternausnetv2': partial(TernausNetV2, num_input_channels=3, num_classes=1),
        'fpn128_wider_resnet20': partial(fpn_v1,
                                         encoder=lambda _: E.WiderResnet20A2Encoder(layers=[1, 2, 3, 4, 5]),
                                         fpn_features=128),
    }

    return registry[model_name.lower()]()


def get_optimizer(optimizer_name: str, parameters, lr: float, **kwargs):
    from torch.optim import SGD, Adam

    if optimizer_name.lower() == 'sgd':
        return SGD(parameters, lr, momentum=0.9, nesterov=True, **kwargs)

    if optimizer_name.lower() == 'adam':
        return Adam(parameters, lr, **kwargs)

    raise ValueError("Unsupported optimizer name " + optimizer_name)


def get_loss(loss_name: str, **kwargs):
    if loss_name.lower() == 'bce':
        return BCEWithLogitsLoss(**kwargs)

    if loss_name.lower() == 'focal':
        return L.BinaryFocalLoss(alpha=None, gamma=1.5, **kwargs)

    if loss_name.lower() == 'reduced_focal':
        return L.BinaryFocalLoss(alpha=None, gamma=1.5, reduced=True, **kwargs)

    if loss_name.lower() == 'bce_jaccard':
        return L.JointLoss(first=BCEWithLogitsLoss(), second=L.BinaryJaccardLogLoss(), first_weight=1.0, second_weight=0.5)

    if loss_name.lower() == 'bce_lovasz':
        return L.JointLoss(first=BCEWithLogitsLoss(), second=L.BinaryLovaszLoss(), first_weight=1.0, second_weight=0.5)

    raise KeyError(loss_name)


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


def predict(model: nn.Module, image: np.ndarray, image_size, tta=None, normalize=A.Normalize(), batch_size=1,
            target_key=None,
            activation='sigmoid') -> np.ndarray:
    model.eval()

    if target_key is not None:
        model = PickModelOutput(model, target_key)

    tile_step = (image_size[0] // 2, image_size[1] // 2)

    tile_slicer = ImageSlicer(image.shape, image_size, tile_step, weight='pyramid')
    tile_merger = CudaTileMerger(tile_slicer.target_shape, 1, tile_slicer.weight)
    patches = tile_slicer.split(image)

    transform = A.Compose([
        normalize,
        A.Lambda(image=_tensor_from_rgb_image)
    ])

    if tta == 'fliplr':
        model = TTAWrapper(model, fliplr_image2mask)

    if tta == 'd4':
        model = TTAWrapper(model, d4_image2mask)

    with torch.no_grad():
        data = list({'image': patch, 'coords': np.array(coords, dtype=np.int)} for (patch, coords) in zip(patches, tile_slicer.crops))
        for batch in DataLoader(InMemoryDataset(data, transform), pin_memory=True, batch_size=batch_size):
            image = batch['image'].cuda(non_blocking=True)
            coords = batch['coords']
            mask_batch = model(image)
            tile_merger.integrate_batch(mask_batch, coords)

    mask = tile_merger.merge()
    if activation == 'sigmoid':
        mask = mask.sigmoid()

    if isinstance(activation, float):
        mask = F.relu(mask_batch - activation, inplace=True)

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


def visualize_inria_predictions(input: dict, output: dict, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    images = []
    for image, target, image_id, logits in zip(input['features'], input['targets'], input['image_id'], output['logits']):
        image = rgb_image_from_tensor(image, mean, std)
        target = to_numpy(target).squeeze(0)
        logits = to_numpy(logits).squeeze(0)

        overlay = np.zeros_like(image)
        true_mask = target > 0
        pred_mask = logits > 0

        overlay[true_mask & pred_mask] = np.array([0, 250, 0], dtype=overlay.dtype)  # Correct predictions (Hits) painted with green
        overlay[true_mask & ~pred_mask] = np.array([250, 0, 0], dtype=overlay.dtype)  # Misses painted with red
        overlay[~true_mask & pred_mask] = np.array([250, 250, 0], dtype=overlay.dtype)  # False alarm painted with yellow

        overlay = cv2.addWeighted(image, 0.5, overlay, 0.5, 0, dtype=cv2.CV_8U)
        cv2.putText(overlay, str(image_id), (10, 15), cv2.FONT_HERSHEY_PLAIN, 1, (250, 250, 250))

        images.append(overlay)
    return images
