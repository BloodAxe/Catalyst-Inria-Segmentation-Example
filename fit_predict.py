from __future__ import absolute_import

import argparse
import collections
import json
import os
from datetime import datetime
from functools import partial
from typing import List, Tuple, Dict

import catalyst
import cv2
import numpy as np
import torch
from catalyst.contrib.nn import OneCycleLRWithWarmup
from catalyst.data import DistributedSamplerWrapper
from catalyst.dl import (
    SupervisedRunner,
    CriterionCallback,
    OptimizerCallback,
    SchedulerCallback,
    MetricAggregationCallback,
    Callback,
)
from catalyst.utils import load_checkpoint, unpack_checkpoint
from pytorch_toolbelt.optimization.functional import get_optimizable_parameters
from pytorch_toolbelt.utils import fs
from pytorch_toolbelt.utils.catalyst import (
    ShowPolarBatchesCallback,
    HyperParametersCallback,
    BestMetricCheckpointCallback,
    PixelAccuracyCallback,
    report_checkpoint,
    clean_checkpoint,
)
from pytorch_toolbelt.utils.random import set_manual_seed
from pytorch_toolbelt.utils.torch_utils import count_parameters, transfer_weights
from sklearn.utils import compute_sample_weight
from torch import nn
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader, WeightedRandomSampler, DistributedSampler

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
    INPUT_IMAGE_ID_KEY,
    get_xview2_extra_dataset,
    INPUT_MASK_WEIGHT_KEY,
    OUTPUT_MASK_2_KEY,
)
from inria.factory import predict
from inria.losses import get_loss, ResizeTargetToPrediction2d
from inria.metric import JaccardMetricPerImageWithOptimalThreshold
from inria.models import get_model
from inria.optim import get_optimizer
from inria.pseudo import BCEOnlinePseudolabelingCallback2d
from inria.scheduler import get_scheduler
from inria.visualization import draw_inria_predictions


def get_criterions(
    criterions,
    criterions_stride2=None,
    criterions_stride4=None,
    criterions_stride8=None,
    criterions_stride16=None,
    ignore_index=None,
) -> Tuple[List[Callback], Dict]:
    criterions_dict = {}
    losses = []
    callbacks = []

    # Create main losses
    for loss_name, loss_weight in criterions:
        criterion_callback = CriterionCallback(
            prefix=f"{OUTPUT_MASK_KEY}/" + loss_name,
            input_key=INPUT_MASK_KEY if loss_name != "wbce" else [INPUT_MASK_KEY, INPUT_MASK_WEIGHT_KEY],
            output_key=OUTPUT_MASK_KEY,
            criterion_key=f"{OUTPUT_MASK_KEY}/" + loss_name,
            multiplier=float(loss_weight),
        )

        criterions_dict[criterion_callback.criterion_key] = get_loss(loss_name, ignore_index=ignore_index)
        callbacks.append(criterion_callback)
        losses.append(criterion_callback.prefix)
        print("Using loss", loss_name, loss_weight)

    # Additional supervision losses
    for supervision_losses, supervision_output in zip(
        [criterions_stride2, criterions_stride4, criterions_stride8, criterions_stride16],
        [OUTPUT_MASK_2_KEY, OUTPUT_MASK_4_KEY, OUTPUT_MASK_8_KEY, OUTPUT_MASK_16_KEY],
    ):
        if supervision_losses is not None:
            for loss_name, loss_weight in supervision_losses:
                prefix = f"{supervision_output}/" + loss_name
                criterion_callback = CriterionCallback(
                    prefix=prefix,
                    input_key=INPUT_MASK_KEY if loss_name != "wbce" else [INPUT_MASK_KEY, INPUT_MASK_WEIGHT_KEY],
                    output_key=supervision_output,
                    criterion_key=prefix,
                    multiplier=float(loss_weight),
                )

                criterions_dict[criterion_callback.criterion_key] = ResizeTargetToPrediction2d(
                    get_loss(loss_name, ignore_index=ignore_index)
                )
                callbacks.append(criterion_callback)
                losses.append(criterion_callback.prefix)
                print("Using loss", loss_name, loss_weight)

    callbacks.append(MetricAggregationCallback(prefix="loss", metrics=losses, mode="sum"))
    return callbacks, criterions_dict


def main():
    parser = argparse.ArgumentParser()

    ###########################################################################################
    # Distributed-training related stuff
    parser.add_argument("--local_rank", type=int, default=0)
    ###########################################################################################

    parser.add_argument("-acc", "--accumulation-steps", type=int, default=1, help="Number of batches to process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument(
        "-dd",
        "--data-dir",
        type=str,
        help="Data directory for INRIA sattelite dataset",
        default=os.environ.get("INRIA_DATA_DIR"),
    )
    parser.add_argument(
        "-dd-xview2", "--data-dir-xview2", type=str, required=False, help="Data directory for external xView2 dataset"
    )
    parser.add_argument("-m", "--model", type=str, default="resnet34_fpncat128", help="")
    parser.add_argument("-b", "--batch-size", type=int, default=8, help="Batch Size during training, e.g. -b 64")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="Epoch to run")
    # parser.add_argument('-es', '--early-stopping', type=int, default=None, help='Maximum number of epochs without improvement')
    # parser.add_argument('-fe', '--freeze-encoder', type=int, default=0, help='Freeze encoder parameters for N epochs')
    # parser.add_argument('-ft', '--fine-tune', action='store_true')
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("-l", "--criterion", type=str, required=True, action="append", nargs="+", help="Criterion")
    parser.add_argument(
        "-l2",
        "--criterion2",
        type=str,
        required=False,
        action="append",
        nargs="+",
        help="Criterion for stride 2 mask",
    )
    parser.add_argument(
        "-l4",
        "--criterion4",
        type=str,
        required=False,
        action="append",
        nargs="+",
        help="Criterion for stride 4 mask",
    )
    parser.add_argument(
        "-l8",
        "--criterion8",
        type=str,
        required=False,
        action="append",
        nargs="+",
        help="Criterion for stride 8 mask",
    )
    parser.add_argument(
        "-l16",
        "--criterion16",
        type=str,
        required=False,
        action="append",
        nargs="+",
        help="Criterion for stride 16 mask",
    )

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
    parser.add_argument("-d", "--dropout", default=None, type=float, help="Dropout before head layer")
    parser.add_argument("--opl", action="store_true")
    parser.add_argument(
        "--warmup", default=0, type=int, help="Number of warmup epochs with reduced LR on encoder parameters"
    )
    parser.add_argument("-wd", "--weight-decay", default=0, type=float, help="L2 weight decay")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--dsv", action="store_true")

    args = parser.parse_args()

    args.is_master = args.local_rank == 0
    args.distributed = False
    fp16 = args.fp16

    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
        args.world_size = int(os.environ["WORLD_SIZE"])
        # args.world_size = torch.distributed.get_world_size()

        print("Initializing init_process_group", args.local_rank)

        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        print("Initialized init_process_group", args.local_rank)

    is_master = args.is_master | (not args.distributed)

    if args.distributed:
        distributed_params = {"rank": args.local_rank, "syncbn": True}
        if args.fp16:
            distributed_params["amp"] = True
    else:
        if args.fp16:
            distributed_params = {}
            distributed_params["amp"] = True
        else:
            distributed_params = False

    set_manual_seed(args.seed + args.local_rank)
    catalyst.utils.set_global_seed(args.seed + args.local_rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data_dir = args.data_dir
    if data_dir is None:
        raise ValueError("--data-dir must be set")

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
    scheduler_name = args.scheduler
    experiment = args.experiment
    dropout = args.dropout
    online_pseudolabeling = args.opl
    criterions = args.criterion
    criterions2 = args.criterion2
    criterions4 = args.criterion4
    criterions8 = args.criterion8
    criterions16 = args.criterion16

    verbose = args.verbose
    show = args.show
    accumulation_steps = args.accumulation_steps
    weight_decay = args.weight_decay
    extra_data_xview2 = args.data_dir_xview2

    run_train = num_epochs > 0
    need_weight_mask = any(c[0] == "wbce" for c in criterions)

    custom_model_kwargs = {}
    if dropout is not None:
        custom_model_kwargs["dropout"] = float(dropout)

    if any([criterions2, criterions4, criterions8, criterions16]):
        custom_model_kwargs["need_supervision_masks"] = True
        print("Enabling supervision masks")

    model: nn.Module = get_model(model_name, **custom_model_kwargs).cuda()

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

    main_metric = "optimized_jaccard"

    current_time = datetime.now().strftime("%y%m%d_%H_%M")
    checkpoint_prefix = f"{current_time}_{args.model}"

    if fp16:
        checkpoint_prefix += "_fp16"

    if fast:
        checkpoint_prefix += "_fast"

    if online_pseudolabeling:
        checkpoint_prefix += "_opl"

    if extra_data_xview2:
        checkpoint_prefix += "_with_xview2"

    if experiment is not None:
        checkpoint_prefix = experiment

    default_callbacks = [
        PixelAccuracyCallback(input_key=INPUT_MASK_KEY, output_key=OUTPUT_MASK_KEY),
        # JaccardMetricPerImage(input_key=INPUT_MASK_KEY, output_key=OUTPUT_MASK_KEY, prefix="jaccard"),
        JaccardMetricPerImageWithOptimalThreshold(
            input_key=INPUT_MASK_KEY, output_key=OUTPUT_MASK_KEY, prefix="optimized_jaccard"
        ),
    ]

    if is_master:

        default_callbacks += [
            BestMetricCheckpointCallback(target_metric="optimized_jaccard", target_metric_minimize=False),
            HyperParametersCallback(
                hparam_dict={
                    "model": model_name,
                    "scheduler": scheduler_name,
                    "optimizer": optimizer_name,
                    "augmentations": augmentations,
                    "size": args.size,
                    "weight_decay": weight_decay,
                    "epochs": num_epochs,
                    "dropout": None if dropout is None else float(dropout),
                }
            ),
        ]

        if show:
            visualize_inria_predictions = partial(
                draw_inria_predictions,
                image_key=INPUT_IMAGE_KEY,
                image_id_key=INPUT_IMAGE_ID_KEY,
                targets_key=INPUT_MASK_KEY,
                outputs_key=OUTPUT_MASK_KEY,
                max_images=16,
            )
            default_callbacks += [
                ShowPolarBatchesCallback(visualize_inria_predictions, metric="accuracy", minimize=False),
                ShowPolarBatchesCallback(visualize_inria_predictions, metric="loss", minimize=True),
            ]

    train_ds, valid_ds, train_sampler = get_datasets(
        data_dir=data_dir,
        image_size=image_size,
        augmentation=augmentations,
        train_mode=train_mode,
        buildings_only=(train_mode == "tiles"),
        fast=fast,
        need_weight_mask=need_weight_mask,
    )

    if extra_data_xview2 is not None:
        extra_train_ds, _ = get_xview2_extra_dataset(
            extra_data_xview2,
            image_size=image_size,
            augmentation=augmentations,
            fast=fast,
            need_weight_mask=need_weight_mask,
        )

        weights = compute_sample_weight("balanced", [0] * len(train_ds) + [1] * len(extra_train_ds))
        train_sampler = WeightedRandomSampler(weights, train_sampler.num_samples * 2)

        train_ds = train_ds + extra_train_ds
        print("Using extra data from xView2 with", len(extra_train_ds), "samples")

    if run_train:
        loaders = collections.OrderedDict()
        callbacks = default_callbacks.copy()
        criterions_dict = {}
        losses = []

        ignore_index = None
        if online_pseudolabeling:
            ignore_index = UNLABELED_SAMPLE
            unlabeled_label = get_pseudolabeling_dataset(
                data_dir, include_masks=False, augmentation=None, image_size=image_size
            )

            unlabeled_train = get_pseudolabeling_dataset(
                data_dir, include_masks=True, augmentation=augmentations, image_size=image_size
            )

            if args.distributed:
                label_sampler = DistributedSampler(unlabeled_label, args.world_size, args.local_rank, shuffle=False)
            else:
                label_sampler = None

            loaders["infer"] = DataLoader(
                unlabeled_label,
                batch_size=batch_size // 2,
                num_workers=num_workers,
                pin_memory=True,
                sampler=label_sampler,
                drop_last=False,
            )

            if train_sampler is not None:
                num_samples = 2 * train_sampler.num_samples
            else:
                num_samples = 2 * len(train_ds)
            weights = compute_sample_weight("balanced", [0] * len(train_ds) + [1] * len(unlabeled_label))

            train_sampler = WeightedRandomSampler(weights, num_samples, replacement=True)
            train_ds = train_ds + unlabeled_train

            callbacks += [
                BCEOnlinePseudolabelingCallback2d(
                    unlabeled_train,
                    pseudolabel_loader="infer",
                    prob_threshold=0.7,
                    output_key=OUTPUT_MASK_KEY,
                    unlabeled_class=UNLABELED_SAMPLE,
                    label_frequency=5,
                )
            ]

            print("Using online pseudolabeling with ", len(unlabeled_label), "samples")

        valid_sampler = None
        if args.distributed:
            if train_sampler is not None:
                train_sampler = DistributedSamplerWrapper(
                    train_sampler, args.world_size, args.local_rank, shuffle=True
                )
            else:
                train_sampler = DistributedSampler(train_ds, args.world_size, args.local_rank, shuffle=True)
            valid_sampler = DistributedSampler(valid_ds, args.world_size, args.local_rank, shuffle=False)

        loaders["train"] = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=train_sampler is None,
            sampler=train_sampler,
        )

        loaders["valid"] = DataLoader(
            valid_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True, sampler=valid_sampler
        )

        loss_callbacks, loss_criterions = get_criterions(
            criterions, criterions2, criterions4, criterions8, criterions16
        )
        callbacks += loss_callbacks

        optimizer = get_optimizer(
            optimizer_name, get_optimizable_parameters(model), learning_rate, weight_decay=weight_decay
        )
        scheduler = get_scheduler(
            scheduler_name, optimizer, lr=learning_rate, num_epochs=num_epochs, batches_in_epoch=len(loaders["train"])
        )
        if isinstance(scheduler, (CyclicLR, OneCycleLRWithWarmup)):
            callbacks += [SchedulerCallback(mode="batch")]

        log_dir = os.path.join("runs", checkpoint_prefix)

        if is_master:
            os.makedirs(log_dir, exist_ok=False)
            config_fname = os.path.join(log_dir, f"{checkpoint_prefix}.json")
            with open(config_fname, "w") as f:
                train_session_args = vars(args)
                f.write(json.dumps(train_session_args, indent=2))

            print("Train session    :", checkpoint_prefix)
            print("  FP16 mode      :", fp16)
            print("  Fast mode      :", args.fast)
            print("  Train mode     :", train_mode)
            print("  Epochs         :", num_epochs)
            print("  Workers        :", num_workers)
            print("  Data dir       :", data_dir)
            print("  Log dir        :", log_dir)
            print("  Augmentations  :", augmentations)
            print("  Train size     :", "batches", len(loaders["train"]), "dataset", len(train_ds))
            print("  Valid size     :", "batches", len(loaders["valid"]), "dataset", len(valid_ds))
            print("Model            :", model_name)
            print("  Parameters     :", count_parameters(model))
            print("  Image size     :", image_size)
            print("Optimizer        :", optimizer_name)
            print("  Learning rate  :", learning_rate)
            print("  Batch size     :", batch_size)
            print("  Criterion      :", criterions)
            print("  Use weight mask:", need_weight_mask)
            if args.distributed:
                print("Distributed")
                print("  World size     :", args.world_size)
                print("  Local rank     :", args.local_rank)
                print("  Is master      :", args.is_master)

        # model training
        runner = SupervisedRunner(input_key=INPUT_IMAGE_KEY, output_key=None, device="cuda")
        runner.train(
            fp16=distributed_params,
            model=model,
            criterion=loss_criterions,
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
        if is_master:
            best_checkpoint = os.path.join(log_dir, "main", "checkpoints", "best.pth")

            model_checkpoint = os.path.join(log_dir, f"{checkpoint_prefix}.pth")
            clean_checkpoint(best_checkpoint, model_checkpoint)

            unpack_checkpoint(torch.load(model_checkpoint), model=model)

            mask = predict(model, read_inria_image("sample_color.jpg"), image_size=image_size, batch_size=args.batch_size)
            mask = ((mask > 0) * 255).astype(np.uint8)
            name = os.path.join(log_dir, "sample_color.jpg")
            cv2.imwrite(name, mask)



if __name__ == "__main__":
    main()
