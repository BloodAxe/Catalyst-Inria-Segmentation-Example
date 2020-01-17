from __future__ import absolute_import

import argparse
import collections
import gc
import os
from datetime import datetime

import cv2
import numpy as np
import torch
from catalyst.contrib.schedulers import OneCycleLRWithWarmup
from catalyst.dl import (
    SupervisedRunner,
    CriterionCallback,
    OptimizerCallback,
    SchedulerCallback,
)
from catalyst.dl.callbacks import CriterionAggregatorCallback
from catalyst.utils import load_checkpoint, unpack_checkpoint
from pytorch_toolbelt.optimization.functional import get_lr_decay_parameters
from pytorch_toolbelt.utils import fs
from pytorch_toolbelt.utils.catalyst import ShowPolarBatchesCallback, PixelAccuracyCallback
from pytorch_toolbelt.utils.random import set_manual_seed
from pytorch_toolbelt.utils.torch_utils import count_parameters, transfer_weights, get_optimizable_parameters
from torch import nn
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader

from inria.dataset import (
    read_inria_image,
    INPUT_IMAGE_KEY,
    OUTPUT_MASK_KEY,
    INPUT_MASK_KEY,
    get_pseudolabeling_dataset,
    get_datasets,
    UNLABELED_SAMPLE,
    OUTPUT_MASK_8_KEY,
    OUTPUT_MASK_4_KEY,
    OUTPUT_MASK_16_KEY,
    OUTPUT_MASK_32_KEY,
)
from inria.factory import visualize_inria_predictions, predict
from inria.losses import get_loss, AdaptiveMaskLoss2d
from inria.metric import JaccardMetricPerImage
from inria.models import get_model
from inria.optim import get_optimizer
from inria.pseudo import BCEOnlinePseudolabelingCallback2d
from inria.scheduler import get_scheduler
from inria.utils import clean_checkpoint, report_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-acc", "--accumulation-steps", type=int, default=1, help="Number of batches to process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument(
        "-dd", "--data-dir", type=str, required=True, help="Data directory for INRIA sattelite dataset"
    )
    parser.add_argument("-m", "--model", type=str, default="resnet34_fpncat128", help="")
    parser.add_argument("-b", "--batch-size", type=int, default=8, help="Batch Size during training, e.g. -b 64")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="Epoch to run")
    # parser.add_argument('-es', '--early-stopping', type=int, default=None, help='Maximum number of epochs without improvement')
    # parser.add_argument('-fe', '--freeze-encoder', type=int, default=0, help='Freeze encoder parameters for N epochs')
    # parser.add_argument('-ft', '--fine-tune', action='store_true')
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("-l", "--criterion", type=str, required=True, action="append", nargs="+", help="Criterion")
    parser.add_argument("-o", "--optimizer", default="RAdam", help="Name of the optimizer")
    parser.add_argument(
        "-c", "--checkpoint", type=str, default=None, help="Checkpoint filename to use as initial model weights"
    )
    parser.add_argument("-w", "--workers", default=8, type=int, help="Num workers")
    parser.add_argument("-a", "--augmentations", default="hard", type=str, help="")
    parser.add_argument("-tm", "--train-mode", default="random", type=str, help="")
    parser.add_argument("--run-mode", default="fit_predict", type=str, help="")
    parser.add_argument("--transfer", default=None, type=str, help="")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--size", default=512, type=int)
    parser.add_argument("-s", "--scheduler", default="multistep", type=str, help="")
    parser.add_argument("-x", "--experiment", default=None, type=str, help="")
    parser.add_argument("-d", "--dropout", default=0.0, type=float, help="Dropout before head layer")
    parser.add_argument("--opl", action="store_true")
    parser.add_argument(
        "--warmup", default=0, type=int, help="Number of warmup epochs with reduced LR on encoder parameters"
    )
    parser.add_argument(
        "-wd", "--weight-decay", default=0, type=float, help="L2 weight decay"
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--dsv", action="store_true")

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
    fp16 = args.fp16
    scheduler_name = args.scheduler
    experiment = args.experiment
    dropout = args.dropout
    online_pseudolabeling = args.opl
    criterions = args.criterion
    verbose = args.verbose
    warmup = args.warmup
    show = args.show
    use_dsv = args.dsv
    accumulation_steps = args.accumulation_steps
    weight_decay = args.weight_decay

    run_train = num_epochs > 0

    model: nn.Module = get_model(model_name, dropout=dropout).cuda()

    if args.transfer:
        transfer_checkpoint = fs.auto_file(args.transfer)
        print("Transfering weights from model checkpoint", transfer_checkpoint)
        checkpoint = load_checkpoint(transfer_checkpoint)
        pretrained_dict = checkpoint["model_state_dict"]

        transfer_weights(model, pretrained_dict)

    if args.checkpoint:
        checkpoint = load_checkpoint(fs.auto_file(args.checkpoint))
        unpack_checkpoint(checkpoint, model=model)

        print("Loaded model weights from:", args.checkpoint)
        report_checkpoint(checkpoint)

    runner = SupervisedRunner(input_key=INPUT_IMAGE_KEY, output_key=None, device="cuda")
    main_metric = "jaccard"
    cmd_args = vars(args)

    current_time = datetime.now().strftime("%b%d_%H_%M")
    checkpoint_prefix = f"{current_time}_{args.model}"

    if fp16:
        checkpoint_prefix += "_fp16"

    if fast:
        checkpoint_prefix += "_fast"

    if online_pseudolabeling:
        checkpoint_prefix += "_opl"

    if experiment is not None:
        checkpoint_prefix = experiment

    log_dir = os.path.join("runs", checkpoint_prefix)
    os.makedirs(log_dir, exist_ok=False)

    default_callbacks = [
        PixelAccuracyCallback(input_key=INPUT_MASK_KEY, output_key=OUTPUT_MASK_KEY),
        JaccardMetricPerImage(input_key=INPUT_MASK_KEY, output_key=OUTPUT_MASK_KEY),
    ]

    if show:
        default_callbacks += [ShowPolarBatchesCallback(visualize_inria_predictions, metric="accuracy", minimize=False)]

    train_ds, valid_ds, train_sampler = get_datasets(
        data_dir=data_dir, image_size=image_size, augmentation=augmentations, train_mode=train_mode, fast=fast
    )

    # Pretrain/warmup
    if warmup:
        callbacks = default_callbacks.copy()
        criterions_dict = {}
        losses = []
        ignore_index = None

        for loss_name, loss_weight in criterions:
            criterion_callback = CriterionCallback(
                prefix="seg_loss/" + loss_name,
                input_key=INPUT_MASK_KEY,
                output_key=OUTPUT_MASK_KEY,
                criterion_key=loss_name,
                multiplier=float(loss_weight),
            )

            criterions_dict[loss_name] = get_loss(loss_name, ignore_index=ignore_index)
            callbacks.append(criterion_callback)
            losses.append(criterion_callback.prefix)
            print("Using loss", loss_name, loss_weight)

        callbacks += [
            CriterionAggregatorCallback(prefix="loss", loss_keys=losses),
            OptimizerCallback(accumulation_steps=accumulation_steps, decouple_weight_decay=False),
        ]

        parameters = get_lr_decay_parameters(model.named_parameters(), learning_rate, {"encoder": 0.1})
        optimizer = get_optimizer("RAdam", parameters, learning_rate=learning_rate * 0.1)

        loaders = collections.OrderedDict()
        loaders["train"] = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=train_sampler is None,
            sampler=train_sampler,
        )

        loaders["valid"] = DataLoader(valid_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

        runner.train(
            fp16=fp16,
            model=model,
            criterion=criterions_dict,
            optimizer=optimizer,
            scheduler=None,
            callbacks=callbacks,
            loaders=loaders,
            logdir=os.path.join(log_dir, "warmup"),
            num_epochs=warmup,
            verbose=verbose,
            main_metric=main_metric,
            minimize_metric=False,
            checkpoint_data={"cmd_args": cmd_args},
        )

        del optimizer, loaders

        best_checkpoint = os.path.join(log_dir, "warmup", "checkpoints", "best.pth")
        model_checkpoint = os.path.join(log_dir, "warmup", "checkpoints", f"{checkpoint_prefix}_warmup.pth")
        clean_checkpoint(best_checkpoint, model_checkpoint)

        torch.cuda.empty_cache()
        gc.collect()

    if run_train:
        loaders = collections.OrderedDict()
        callbacks = default_callbacks.copy()
        criterions_dict = {}
        losses = []

        ignore_index = None
        if online_pseudolabeling:
            ignore_index = UNLABELED_SAMPLE
            unlabeled_label = get_pseudolabeling_dataset(data_dir, include_masks=False, image_size=image_size)

            unlabeled_train = get_pseudolabeling_dataset(
                data_dir, include_masks=True, image_size=image_size, augmentation=augmentations
            )

            loaders["label"] = DataLoader(
                unlabeled_label, batch_size=batch_size, num_workers=num_workers, pin_memory=True
            )

            # if train_sampler is not None:
            #     num_samples = 2 * train_sampler.num_samples
            # else:
            #     num_samples = 2 * len(train_ds)
            # weights = compute_sample_weight("balanced", [0] * len(train_ds) + [1] * len(unlabeled_label))

            # train_sampler = WeightedRandomSampler(weights, num_samples, replacement=True)
            train_ds = train_ds + unlabeled_train
            train_sampler = None
            
            callbacks += [
                BCEOnlinePseudolabelingCallback2d(
                    unlabeled_train,
                    pseudolabel_loader="label",
                    prob_threshold=0.75,
                    output_key=OUTPUT_MASK_KEY,
                    unlabeled_class=UNLABELED_SAMPLE,
                    label_frequency=5,
                )
            ]

            print("Using online pseudolabeling with ", len(unlabeled_label), "samples")

        loaders["train"] = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=train_sampler is None,
            sampler=train_sampler,
        )

        loaders["valid"] = DataLoader(valid_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

        # Create losses
        for loss_name, loss_weight in criterions:
            criterion_callback = CriterionCallback(
                prefix="seg_loss/" + loss_name,
                input_key=INPUT_MASK_KEY,
                output_key=OUTPUT_MASK_KEY,
                criterion_key=loss_name,
                multiplier=float(loss_weight),
            )

            criterions_dict[loss_name] = get_loss(loss_name, ignore_index=ignore_index)
            callbacks.append(criterion_callback)
            losses.append(criterion_callback.prefix)
            print("Using loss", loss_name, loss_weight)

        if use_dsv:
            print("Using DSV")
            criterions = "dsv"
            dsv_loss_name = "soft_bce"

            criterions_dict[criterions] = AdaptiveMaskLoss2d(get_loss(dsv_loss_name, ignore_index=ignore_index))

            for i, dsv_input in enumerate(
                [OUTPUT_MASK_4_KEY, OUTPUT_MASK_8_KEY, OUTPUT_MASK_16_KEY, OUTPUT_MASK_32_KEY]
            ):
                criterion_callback = CriterionCallback(
                    prefix="seg_loss_dsv/" + dsv_input,
                    input_key=OUTPUT_MASK_KEY,
                    output_key=dsv_input,
                    criterion_key=criterions,
                    multiplier=1.0,
                )
                callbacks.append(criterion_callback)
                losses.append(criterion_callback.prefix)

        callbacks += [
            CriterionAggregatorCallback(prefix="loss", loss_keys=losses),
            OptimizerCallback(accumulation_steps=accumulation_steps, decouple_weight_decay=False),
        ]

        optimizer = get_optimizer(optimizer_name, get_optimizable_parameters(model), learning_rate, weight_decay=weight_decay)
        scheduler = get_scheduler(
            scheduler_name, optimizer, lr=learning_rate, num_epochs=num_epochs, batches_in_epoch=len(loaders["train"])
        )
        if isinstance(scheduler, (CyclicLR, OneCycleLRWithWarmup)):
            callbacks += [SchedulerCallback(mode="batch")]

        print("Train session    :", checkpoint_prefix)
        print("\tFP16 mode      :", fp16)
        print("\tFast mode      :", args.fast)
        print("\tTrain mode     :", train_mode)
        print("\tEpochs         :", num_epochs)
        print("\tWorkers        :", num_workers)
        print("\tData dir       :", data_dir)
        print("\tLog dir        :", log_dir)
        print("\tAugmentations  :", augmentations)
        print("\tTrain size     :", len(loaders["train"]), len(train_ds))
        print("\tValid size     :", len(loaders["valid"]), len(valid_ds))
        print("Model            :", model_name)
        print("\tParameters     :", count_parameters(model))
        print("\tImage size     :", image_size)
        print("Optimizer        :", optimizer_name)
        print("\tLearning rate  :", learning_rate)
        print("\tBatch size     :", batch_size)
        print("\tCriterion      :", criterions)

        # model training
        runner.train(
            fp16=fp16,
            model=model,
            criterion=criterions_dict,
            optimizer=optimizer,
            scheduler=scheduler,
            callbacks=callbacks,
            loaders=loaders,
            logdir=os.path.join(log_dir, "main"),
            num_epochs=num_epochs,
            verbose=verbose,
            main_metric=main_metric,
            minimize_metric=False,
            checkpoint_data={"cmd_args": vars(args)},
        )

        # Training is finished. Let's run predictions using best checkpoint weights
        best_checkpoint = os.path.join(log_dir, "main", "checkpoints", "best.pth")

        model_checkpoint = os.path.join(log_dir, "warmup", "checkpoints", f"{checkpoint_prefix}.pth")
        clean_checkpoint(best_checkpoint, model_checkpoint)

        unpack_checkpoint(torch.load(model_checkpoint), model=model)

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


if __name__ == "__main__":
    main()
