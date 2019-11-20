import argparse

import cv2, os
from pytorch_toolbelt.inference.tiles import ImageSlicer
from pytorch_toolbelt.utils.fs import id_from_fname, read_image_as_is
import pandas as pd
from tqdm import tqdm

from inria.dataset import TEST_LOCATIONS


def split_image(image_fname, output_dir, tile_size, tile_step, image_margin):
    os.makedirs(output_dir, exist_ok=True)
    image = read_image_as_is(image_fname)
    image_id = id_from_fname(image_fname)

    slicer = ImageSlicer(image.shape, tile_size, tile_step, image_margin)
    tiles = slicer.split(image)

    fnames = []
    for i, tile in enumerate(tiles):
        output_fname = os.path.join(output_dir, f"{image_id}_tile_{i}.png")
        cv2.imwrite(output_fname, tile)
        fnames.append(output_fname)

    return fnames


def cut_dataset_in_patches(data_dir, tile_size, tile_step, image_margin):
    locations = TEST_LOCATIONS

    train_data = []

    # For validation, we remove the first five images of every location (e.g., austin{1-5}.tif, chicago{1-5}.tif) from the training set.
    # That is suggested validation strategy by competition host
    for loc in locations:
        for i in range(1, 37):
            train_data.append(f"{loc}{i}")

    train_imgs = [os.path.join(data_dir, "test", "images", f"{fname}.tif") for fname in train_data]

    images_dir = os.path.join(data_dir, "test_tiles", "images")

    for train_img in tqdm(train_imgs, total=len(train_imgs), desc="test_imgs"):
        split_image(train_img, images_dir, tile_size, tile_step, image_margin)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dd", "--data-dir", type=str, required=True, help="Data directory for INRIA sattelite dataset"
    )
    args = parser.parse_args()

    cut_dataset_in_patches(args.data_dir, tile_size=(512, 512), tile_step=(512, 512), image_margin=0)


if __name__ == "__main__":
    main()
