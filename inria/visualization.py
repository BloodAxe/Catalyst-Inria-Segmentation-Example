from typing import Callable, Optional, List, Union

import cv2
import numpy as np
from pytorch_toolbelt.utils.torch_utils import rgb_image_from_tensor, to_numpy

from inria.dataset import (
    OUTPUT_OFFSET_KEY,
    OUTPUT_MASK_4_KEY,
    OUTPUT_MASK_32_KEY,
    OUTPUT_MASK_16_KEY,
    OUTPUT_MASK_8_KEY,
    OUTPUT_MASK_2_KEY,
)


def draw_inria_predictions(
    input: dict,
    output: dict,
    inputs_to_labels:Callable,
    outputs_to_labels: Callable,
    image_key="features",
    image_id_key: Optional[str] = "image_id",
    targets_key="targets",
    outputs_key="logits",
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    max_images=None,
    image_format: Union[str, Callable] = "bgr",
) -> List[np.ndarray]:
    """
    Render visualization of model's prediction for binary segmentation problem.
    This function draws a color-coded overlay on top of the image, with color codes meaning:
        - green: True positives
        - red: False-negatives
        - yellow: False-positives

    :param input: Input batch (model's input batch)
    :param output: Output batch (model predictions)
    :param image_key: Key for getting image
    :param image_id_key: Key for getting image id/fname
    :param targets_key: Key for getting ground-truth mask
    :param outputs_key: Key for getting model logits for predicted mask
    :param mean: Mean vector user during normalization
    :param std: Std vector user during normalization
    :param max_images: Maximum number of images to visualize from batch
        (If you have huge batch, saving hundreds of images may make TensorBoard slow)
    :param targets_threshold: Threshold to convert target values to binary.
        Default value 0.5 is safe for both smoothed and hard labels.
    :param logits_threshold: Threshold to convert model predictions (raw logits) values to binary.
        Default value 0.0 is equivalent to 0.5 after applying sigmoid activation
    :param image_format: Source format of the image tensor to conver to RGB representation.
        Can be string ("gray", "rgb", "brg") or function `convert(np.ndarray)->nd.ndarray`.
    :return: List of images
    """
    images = []
    num_samples = len(input[image_key])
    if max_images is not None:
        num_samples = min(num_samples, max_images)

    true_masks = to_numpy(inputs_to_labels(input[targets_key])).astype(bool)
    pred_masks = to_numpy(outputs_to_labels(output[outputs_key])).astype(bool)

    for i in range(num_samples):
        image = rgb_image_from_tensor(input[image_key][i], mean, std)

        if image_format == "bgr":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image_format == "gray":
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif hasattr(image_format, "__call__"):
            image = image_format(image)

        overlay = image.copy()
        true_mask = true_masks[i]
        pred_mask = pred_masks[i]

        overlay[true_mask & pred_mask] = np.array(
            [0, 250, 0], dtype=overlay.dtype
        )  # Correct predictions (Hits) painted with green
        overlay[true_mask & ~pred_mask] = np.array([250, 0, 0], dtype=overlay.dtype)  # Misses painted with red
        overlay[~true_mask & pred_mask] = np.array(
            [250, 250, 0], dtype=overlay.dtype
        )  # False alarm painted with yellow
        overlay = cv2.addWeighted(image, 0.5, overlay, 0.5, 0, dtype=cv2.CV_8U)

        if OUTPUT_OFFSET_KEY in output:
            offset = to_numpy(output[OUTPUT_OFFSET_KEY][i]) * 32
            offset = np.expand_dims(offset, -1)

            x = offset[0, ...].clip(min=0, max=1) * np.array([255, 0, 0]) + (-offset[0, ...]).clip(
                min=0, max=1
            ) * np.array([0, 0, 255])
            y = offset[1, ...].clip(min=0, max=1) * np.array([255, 0, 255]) + (-offset[1, ...]).clip(
                min=0, max=1
            ) * np.array([0, 255, 0])

            offset = (x + y).clip(0, 255).astype(np.uint8)
            offset = cv2.resize(offset, (image.shape[1], image.shape[0]))
            overlay = np.row_stack([overlay, offset])

        dsv_inputs = [OUTPUT_MASK_2_KEY, OUTPUT_MASK_4_KEY, OUTPUT_MASK_8_KEY, OUTPUT_MASK_16_KEY, OUTPUT_MASK_32_KEY]
        for dsv_input_key in dsv_inputs:
            if dsv_input_key in output:
                dsv_p = to_numpy(output[dsv_input_key][i].detach().float().sigmoid().squeeze(0))
                dsv_p = cv2.resize((dsv_p * 255).astype(np.uint8), (image.shape[1], image.shape[0]))
                dsv_p = cv2.cvtColor(dsv_p, cv2.COLOR_GRAY2RGB)
                overlay = np.row_stack([overlay, dsv_p])

        if image_id_key is not None and image_id_key in input:
            image_id = input[image_id_key][i]
            cv2.putText(overlay, str(image_id), (10, 15), cv2.FONT_HERSHEY_PLAIN, 1, (250, 250, 250))

        images.append(overlay)
    return images
