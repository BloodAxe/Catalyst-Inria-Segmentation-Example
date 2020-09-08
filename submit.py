import argparse
import os
import subprocess

import cv2
import numpy as np
import torch
from pytorch_toolbelt.inference.tta import TTAWrapper, d4_image2mask, fliplr_image2mask, MultiscaleTTAWrapper
from pytorch_toolbelt.utils.catalyst import report_checkpoint
from torch import nn

from tqdm import tqdm
from pytorch_toolbelt.utils.fs import auto_file, find_in_dir

from inria.dataset import read_inria_image, OUTPUT_MASK_KEY
from inria.factory import predict, PickModelOutput
from inria.models import model_from_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="unet", help="")
    parser.add_argument("-dd", "--data-dir", type=str, default=None, required=True, help="Data dir")
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default=None,
        required=True,
        help="Checkpoint filename to use as initial model weights",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("-tta", "--tta", default=None, type=str, help="Type of TTA to use [fliplr, d4]")
    args = parser.parse_args()

    data_dir = args.data_dir
    checkpoint_file = auto_file(args.checkpoint)
    run_dir = os.path.dirname(checkpoint_file)
    out_dir = os.path.join(run_dir, "submit")
    os.makedirs(out_dir, exist_ok=True)

    model, checkpoint = model_from_checkpoint(checkpoint_file, strict=False)
    threshold = checkpoint["epoch_metrics"].get("valid_optimized_jaccard/threshold", 0.5)
    print(report_checkpoint(checkpoint))
    print("Using threshold", threshold)

    model = nn.Sequential(PickModelOutput(model, OUTPUT_MASK_KEY), nn.Sigmoid())

    if args.tta == "fliplr":
        model = TTAWrapper(model, fliplr_image2mask)
    elif args.tta == "d4":
        model = TTAWrapper(model, d4_image2mask)
    elif args.tta == "ms-d2":
        model = TTAWrapper(model, fliplr_image2mask)
        model = MultiscaleTTAWrapper(model, size_offsets=[-128, -64, 64, 128])
    elif args.tta == "ms-d4":
        model = TTAWrapper(model, d4_image2mask)
        model = MultiscaleTTAWrapper(model, size_offsets=[-128, -64, 64, 128])
    elif args.tta == "ms":
        model = MultiscaleTTAWrapper(model, size_offsets=[-128, -64, 64, 128])
    else:
        pass

    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.eval()

    # mask = predict(model, read_inria_image("sample_color.jpg"), image_size=(512, 512), batch_size=args.batch_size * torch.cuda.device_count())
    # mask = ((mask > threshold) * 255).astype(np.uint8)
    # name = os.path.join(run_dir, "sample_color.jpg")
    # cv2.imwrite(name, mask)

    test_predictions_dir = os.path.join(out_dir, "test_predictions")
    test_predictions_dir_compressed = os.path.join(out_dir, "test_predictions_compressed")

    if args.tta is not None:
        test_predictions_dir += f"_{args.tta}"
        test_predictions_dir_compressed += f"_{args.tta}"

    os.makedirs(test_predictions_dir, exist_ok=True)
    os.makedirs(test_predictions_dir_compressed, exist_ok=True)

    test_images = find_in_dir(os.path.join(data_dir, "test", "images"))
    for fname in tqdm(test_images, total=len(test_images)):
        predicted_mask_fname = os.path.join(test_predictions_dir, os.path.basename(fname))

        if not os.path.isfile(predicted_mask_fname):
            image = read_inria_image(fname)
            mask = predict(model, image, image_size=(512, 512), batch_size=args.batch_size * torch.cuda.device_count())
            mask = ((mask > threshold) * 255).astype(np.uint8)
            cv2.imwrite(predicted_mask_fname, mask)

        name_compressed = os.path.join(test_predictions_dir_compressed, os.path.basename(fname))
        command = (
            "gdal_translate --config GDAL_PAM_ENABLED NO -co COMPRESS=CCITTFAX4 -co NBITS=1 "
            + predicted_mask_fname
            + " "
            + name_compressed
        )
        subprocess.call(command, shell=True)


if __name__ == "__main__":
    # Give no chance to randomness
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main()
