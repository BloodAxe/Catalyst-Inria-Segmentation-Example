import argparse
import os
import cv2
import numpy as np
import torch
from catalyst.utils import load_checkpoint, unpack_checkpoint
from pytorch_toolbelt.inference.tta import TTAWrapper, d4_image2mask, fliplr_image2mask, MultiscaleTTAWrapper
from torch import nn

from tqdm import tqdm
from pytorch_toolbelt.utils.fs import auto_file, find_in_dir

from inria.dataset import read_inria_image, OUTPUT_MASK_KEY
from inria.factory import predict, PickModelOutput
from inria.models import get_model


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
    run_dir = os.path.dirname(os.path.dirname(checkpoint_file))
    out_dir = os.path.join(run_dir, "submit")
    os.makedirs(out_dir, exist_ok=True)

    checkpoint = load_checkpoint(checkpoint_file)
    checkpoint_epoch = checkpoint["epoch"]
    print("Loaded model weights from", args.checkpoint)
    print("Epoch   :", checkpoint_epoch)
    print(
        "Metrics (Train):",
        "IoU:",
        checkpoint["epoch_metrics"]["train"]["jaccard"],
        "Acc:",
        checkpoint["epoch_metrics"]["train"]["accuracy"],
    )
    print(
        "Metrics (Valid):",
        "IoU:",
        checkpoint["epoch_metrics"]["valid"]["jaccard"],
        "Acc:",
        checkpoint["epoch_metrics"]["valid"]["accuracy"],
    )

    model = get_model(args.model)
    unpack_checkpoint(checkpoint, model=model)

    model = nn.Sequential(PickModelOutput(model, OUTPUT_MASK_KEY), nn.Sigmoid())

    if args.tta == "fliplr":
        model = TTAWrapper(model, fliplr_image2mask)

    if args.tta == "flipscale":
        model = TTAWrapper(model, fliplr_image2mask)
        model = MultiscaleTTAWrapper(model, size_offsets=[-128, -64, 64, 128])

    if args.tta == "d4":
        model = TTAWrapper(model, d4_image2mask)

    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.eval()

    mask = predict(model, read_inria_image("sample_color.jpg"), image_size=(512, 512), batch_size=args.batch_size)
    mask = ((mask > 0.5) * 255).astype(np.uint8)
    name = os.path.join(run_dir, "sample_color.jpg")
    cv2.imwrite(name, mask)

    predictions_dir = os.path.join(out_dir, f"test_predictions")
    if args.tta is not None:
        predictions_dir += f"_{args.tta}"

    os.makedirs(predictions_dir, exist_ok=True)

    test_images = find_in_dir(os.path.join(data_dir, "test", "images"))
    for fname in tqdm(test_images, total=len(test_images)):
        image = read_inria_image(fname)
        mask = predict(model, image, image_size=(512, 512), batch_size=args.batch_size)
        mask = ((mask > 0.5) * 255).astype(np.uint8)
        name = os.path.join(predictions_dir, os.path.basename(fname))
        cv2.imwrite(name, mask)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
