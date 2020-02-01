from inria.dataset import compute_weight_mask, read_inria_mask, read_xview_mask
import matplotlib.pyplot as plt


def test_compute_weight_mask():
    mask = read_xview_mask("mask.png")

    w = compute_weight_mask(mask, edge_weight=4)

    plt.figure(figsize=(10, 10))
    plt.tight_layout()
    plt.imshow(w)
    plt.axis("off")
    plt.show()
