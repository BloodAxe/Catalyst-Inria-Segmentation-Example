from inria.dataset import compute_weight_mask, read_inria_mask, read_xview_mask, mask2depth, depth2mask
import matplotlib.pyplot as plt
import numpy as np


def test_compute_weight_mask():
    mask = read_xview_mask("mask.png")

    w = compute_weight_mask(mask, edge_weight=4)

    plt.figure(figsize=(10, 10))
    plt.tight_layout()
    plt.imshow(w)
    plt.axis("off")
    plt.show()


def test_mask2depth():
    x = np.random.randint(0, 2, (512, 512), dtype=np.uint8)
    a = mask2depth(x)
    print(np.bincount(a.flatten(),minlength=16))
    y = depth2mask(a)
    np.testing.assert_equal(x, y)
