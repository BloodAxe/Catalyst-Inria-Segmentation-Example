import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, binary_dilation


def compute_boundary_mask(mask: np.ndarray) -> np.ndarray:
    dilated = binary_dilation(mask, structure=np.ones((5, 5), dtype=np.bool))
    dilated = binary_fill_holes(dilated)

    # dilated = cv2.dilate(mask, kernel=(5, 5))
    # eroded = cv2.erode(dilated, kernel=(5, 5))
    # morphoplogy_open

    diff = dilated & ~mask
    diff = cv2.dilate(diff, kernel=(5, 5))
    diff = diff & ~mask
    return diff.astype(np.uint8)


mask = cv2.imread("d:\\datasets\\inria\\train\\gt\\chicago18.tif", cv2.IMREAD_GRAYSCALE)[:1024, :1024]
edge = compute_boundary_mask(mask)
zero = np.zeros_like(mask)

msk_view = np.dstack(((mask > 0) * 255, (edge > 0) * 255, zero)).astype(np.uint8)

cv2.imshow("Mask", msk_view)
cv2.waitKey(-1)
