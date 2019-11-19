from __future__ import absolute_import

import argparse
import collections
import os
from datetime import datetime

import cv2
import numpy as np
from catalyst.dl import SupervisedRunner, EarlyStoppingCallback, CriterionCallback
from catalyst.utils import load_checkpoint, unpack_checkpoint
from pytorch_toolbelt.utils import fs
from pytorch_toolbelt.utils.catalyst import (
    ShowPolarBatchesCallback,
    PixelAccuracyCallback,
)
from pytorch_toolbelt.utils.random import set_manual_seed
from pytorch_toolbelt.utils.torch_utils import (
    count_parameters,
    transfer_weights,
)
from torch import nn
from tqdm import tqdm

from inria.dataset import (
    get_dataloaders,
    read_inria_image,
    INPUT_IMAGE_KEY,
    OUTPUT_MASK_KEY,
    INPUT_MASK_KEY,
)
from inria.factory import get_loss, visualize_inria_predictions, predict
from inria.metric import JaccardMetricPerImage
from inria.models import get_model
from inria.optim import get_optimizer
from inria.scheduler import get_scheduler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument(
        "-dd",
        "--data-dir",
        type=str,
        required=True,
        help="Data directory for INRIA sattelite dataset",
    )
    parser.add_argument(
        "-m", "--model", type=str, default="resnet34_fpncat128", help=""
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=8,
        help="Batch Size during training, e.g. -b 64",
    )
    parser.add_argument("-e", "--epochs", type=int, default=100, help="Epoch to run")
    # parser.add_argument('-es', '--early-stopping', type=int, default=None, help='Maximum number of epochs without improvement')
    # parser.add_argument('-fe', '--freeze-encoder', type=int, default=0, help='Freeze encoder parameters for N epochs')
    # parser.add_argument('-ft', '--fine-tune', action='store_true')
    parser.add_argument(
        "-lr", "--learning-rate", type=float, default=1e-3, help="Initial learning rate"
    )
    parser.add_argument("-l", "--criterion", type=str, default="bce", help="Criterion")
    parser.add_argument(
        "-o", "--optimizer", default="Adam", help="Name of the optimizer"
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint filename to use as initial model weights",
    )
    parser.add_argument("-w", "--workers", default=8, type=int, help="Num workers")
    parser.add_argument("-a", "--augmentations", default="hard", type=str, help="")
    parser.add_argument(
        "-tta", "--tta", default=None, type=str, help="Type of TTA to use [fliplr, d4]"
    )
    parser.add_argument("-tm", "--train-mode", default="random", type=str, help="")
    parser.add_argument("--run-mode", default="fit_predict", type=str, help="")
    parser.add_argument("--transfer", default=None, type=str, help="")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--size", default=512, type=int)
    parser.add_argument("-s", "--scheduler", default="multistep", type=str, help="")
    parser.add_argument("-x", "--experiment", default=None, type=str, help="")
    parser.add_argument(
        "-d", "--dropout", default=0.0, type=float, help="Dropout before head layer"
    )

    args = parser.parse_args()
    set_manual_seed(args.seed)

    data_dir = args.data_dir
    num_workers = args.workers
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    model_name = args.model
    optimizer_name = args.optimizer
    image_size = args.size, args.size
    fast = args.fast
    augmentations = args.augmentations
    train_mode = args.train_mode
    run_mode = args.run_mode
    log_dir = None
    fp16 = args.fp16
    scheduler_name = args.scheduler
    experiment = args.experiment
    dropout = args.dropout

    run_train = run_mode == "fit_predict" or run_mode == "fit"
    run_predict = run_mode == "fit_predict" or run_mode == "predict"

    model: nn.Module = get_model(model_name, dropout=dropout).cuda()

    if args.transfer:
        transfer_checkpoint = fs.auto_file(args.transfer)
        print("Transfering weights from model checkpoint", transfer_checkpoint)
        checkpoint = load_checkpoint(transfer_checkpoint)
        pretrained_dict = checkpoint["model_state_dict"]

        transfer_weights(model, pretrained_dict)

    checkpoint = None
    if args.checkpoint:
        checkpoint = load_checkpoint(fs.auto_file(args.checkpoint))
        unpack_checkpoint(checkpoint, model=model)

        checkpoint_epoch = checkpoint["epoch"]
        print("Loaded model weights from:", args.checkpoint)
        print("Epoch                    :", checkpoint_epoch)
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

        log_dir = os.path.dirname(os.path.dirname(fs.auto_file(args.checkpoint)))

    if run_train:
        criterion = get_loss(args.criterion)
        optimizer = get_optimizer(optimizer_name, model.parameters(), learning_rate)

        train_loader, valid_loader = get_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
            augmentation=augmentations,
            train_mode=train_mode,
            fast=fast,
        )

        loaders = collections.OrderedDict()
        loaders["train"] = train_loader
        loaders["valid"] = valid_loader

        current_time = datetime.now().strftime("%b%d_%H_%M")
        checkpoint_prefix = f"{current_time}_{args.model}_edge_{args.criterion}"

        if fp16:
            checkpoint_prefix += "_fp16"

        if fast:
            checkpoint_prefix += "_fast"

        if experiment is not None:
            checkpoint_prefix = experiment

        log_dir = os.path.join("runs", checkpoint_prefix)
        os.makedirs(log_dir, exist_ok=False)

        scheduler = get_scheduler(
            scheduler_name,
            optimizer,
            lr=learning_rate,
            num_epochs=num_epochs,
            batches_in_epoch=len(train_loader),
        )
        criterion_callback = CriterionCallback(
            input_key=INPUT_MASK_KEY, output_key=OUTPUT_MASK_KEY,
        )

        # model runner
        runner = SupervisedRunner(input_key=INPUT_IMAGE_KEY, output_key=None)

        print("Train session    :", checkpoint_prefix)
        print("\tFP16 mode      :", fp16)
        print("\tFast mode      :", args.fast)
        print("\tTrain mode     :", train_mode)
        print("\tEpochs         :", num_epochs)
        print("\tWorkers        :", num_workers)
        print("\tData dir       :", data_dir)
        print("\tLog dir        :", log_dir)
        print("\tAugmentations  :", augmentations)
        print("\tTrain size     :", len(train_loader), len(train_loader.dataset))
        print("\tValid size     :", len(valid_loader), len(valid_loader.dataset))
        print("Model            :", model_name)
        print("\tParameters     :", count_parameters(model))
        print("\tImage size     :", image_size)
        print("Optimizer        :", optimizer_name)
        print("\tLearning rate  :", learning_rate)
        print("\tBatch size     :", batch_size)
        print("\tCriterion      :", args.criterion)

        # model training
        runner.train(
            fp16=fp16,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            callbacks=[
                criterion_callback,
                PixelAccuracyCallback(
                    input_key=INPUT_MASK_KEY, output_key=OUTPUT_MASK_KEY
                ),
                JaccardMetricPerImage(
                    input_key=INPUT_MASK_KEY, output_key=OUTPUT_MASK_KEY
                ),
                # OptimalThreshold(),
                ShowPolarBatchesCallback(
                    visualize_inria_predictions, metric="accuracy", minimize=False
                ),
                EarlyStoppingCallback(30, metric="jaccard", minimize=False),
            ],
            loaders=loaders,
            logdir=log_dir,
            num_epochs=num_epochs,
            verbose=True,
            main_metric="jaccard",
            minimize_metric=False,
            state_kwargs={"cmd_args": vars(args)},
        )

        # Training is finished. Let's run predictions using best checkpoint weights
        best_checkpoint = load_checkpoint(fs.auto_file("best.pth", where=log_dir))
        unpack_checkpoint(best_checkpoint, model=model)

        mask = predict(
            model,
            read_inria_image("sample_color.jpg"),
            tta=args.tta,
            image_size=image_size,
            target_key=OUTPUT_MASK_KEY,
            batch_size=args.batch_size,
            activation="sigmoid",
        )
        mask = ((mask > 0.5) * 255).astype(np.uint8)
        name = os.path.join(log_dir, "sample_color.jpg")
        cv2.imwrite(name, mask)

        del optimizer, loaders

    if run_predict and not fast:

        mask = predict(
            model,
            read_inria_image("sample_color.jpg"),
            tta=args.tta,
            image_size=image_size,
            target_key=OUTPUT_MASK_KEY,
            batch_size=args.batch_size,
            activation="sigmoid",
        )
        mask = ((mask > 0.5) * 255).astype(np.uint8)
        name = os.path.join(log_dir, "sample_color.jpg")
        cv2.imwrite(name, mask)

        out_dir = os.path.join(log_dir, "submit")
        os.makedirs(out_dir, exist_ok=True)

        test_images = fs.find_in_dir(os.path.join(data_dir, "test", "images"))
        for fname in tqdm(test_images, total=len(test_images)):
            image = read_inria_image(fname)
            mask = predict(
                model,
                image,
                tta=args.tta,
                image_size=image_size,
                batch_size=args.batch_size,
                target_key=OUTPUT_MASK_KEY,
                activation="sigmoid",
            )
            mask = ((mask > 0.5) * 255).astype(np.uint8)
            name = os.path.join(out_dir, os.path.basename(fname))
            cv2.imwrite(name, mask)


if __name__ == "__main__":
    main()
