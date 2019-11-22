import argparse

import cv2, os
from pytorch_toolbelt.inference.tiles import ImageSlicer
from pytorch_toolbelt.utils.fs import id_from_fname, read_image_as_is
import pandas as pd
from tqdm import tqdm


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
    locations = ["austin", "chicago", "kitsap", "tyrol-w", "vienna"]

    train_data = []
    valid_data = []

    # For validation, we remove the first five images of every location (e.g., austin{1-5}.tif, chicago{1-5}.tif) from the training set.
    # That is suggested validation strategy by competition host
    for loc in locations:
        for i in range(1, 6):
            valid_data.append(f"{loc}{i}")
        for i in range(6, 37):
            train_data.append(f"{loc}{i}")

    train_imgs = [os.path.join(data_dir, "train", "images", f"{fname}.tif") for fname in train_data]
    valid_imgs = [os.path.join(data_dir, "train", "images", f"{fname}.tif") for fname in valid_data]

    train_masks = [os.path.join(data_dir, "train", "gt", f"{fname}.tif") for fname in train_data]
    valid_masks = [os.path.join(data_dir, "train", "gt", f"{fname}.tif") for fname in valid_data]

    images_dir = os.path.join(data_dir, "train_tiles", "images")
    masks_dir = os.path.join(data_dir, "train_tiles", "gt")

    all_image_tiles = []
    all_mask_tiles = []
    train_flag = []

    for train_img in tqdm(train_imgs, total=len(train_imgs), desc="train_imgs"):
        img_tiles = split_image(train_img, images_dir, tile_size, tile_step, image_margin)
        all_image_tiles.extend(img_tiles)
        train_flag.extend([1] * len(img_tiles))

    for train_msk in tqdm(train_masks, total=len(train_masks), desc="train_masks"):
        msk_tiles = split_image(train_msk, masks_dir, tile_size, tile_step, image_margin)
        all_mask_tiles.extend(msk_tiles)

    for valid_img in tqdm(valid_imgs, total=len(valid_imgs), desc="valid_imgs"):
        img_tiles = split_image(valid_img, images_dir, tile_size, tile_size, image_margin)
        all_image_tiles.extend(img_tiles)
        train_flag.extend([0] * len(img_tiles))

    for valid_msk in tqdm(valid_masks, total=len(valid_masks), desc="valid_masks"):
        msk_tiles = split_image(valid_msk, masks_dir, tile_size, tile_size, image_margin)
        all_mask_tiles.extend(msk_tiles)

    return pd.DataFrame.from_dict({"image": all_image_tiles, "mask": all_mask_tiles, "train": train_flag})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dd", "--data-dir", type=str, required=True, help="Data directory for INRIA sattelite dataset"
    )
    args = parser.parse_args()

    df = cut_dataset_in_patches(args.data_dir, tile_size=(512, 512), tile_step=(384, 384), image_margin=0)
    df.to_csv(os.path.join(args.data_dir, "inria_tiles.csv"))


if __name__ == "__main__":
    main()
