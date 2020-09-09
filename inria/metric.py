from collections import defaultdict
from typing import Callable

import numpy as np
import torch
from pytorch_toolbelt.utils import to_numpy
from pytorch_toolbelt.utils.catalyst import get_tensorboard_logger
from pytorch_toolbelt.utils.distributed import all_gather
from catalyst.core import Callback, CallbackNode, CallbackOrder, IRunner
from catalyst.dl import registry
from catalyst.utils.distributed import get_rank

__all__ = ["JaccardMetricPerImage", "JaccardMetricPerImageWithOptimalThreshold"]


@registry.Callback
class JaccardMetricPerImage(Callback):
    """
    Jaccard metric callback which computes IoU metric per image and is aware that image is tiled.
    """

    def __init__(
        self,
        inputs_to_labels: Callable,
        outputs_to_labels: Callable,
        input_key: str = "targets",
        output_key: str = "logits",
        image_id_key: str = "image_id",
        prefix: str = "jaccard",
    ):
        super().__init__(CallbackOrder.Metric, CallbackNode.All)
        """
        :param input_key: input key to use for precision calculation; specifies our `y_true`.
        :param output_key: output key to use for precision calculation; specifies our `y_pred`.
        """
        self.prefix = prefix
        self.output_key = output_key
        self.input_key = input_key
        self.image_id_key = image_id_key
        self.scores_per_image = {}
        self.locations = ["austin", "chicago", "kitsap", "tyrol-w", "vienna"]
        self.inputs_to_labels = inputs_to_labels
        self.outputs_to_labels = outputs_to_labels

    def on_loader_start(self, state):
        self.scores_per_image = {}

    def on_batch_end(self, runner: IRunner):
        image_ids = runner.input[self.image_id_key]
        outputs = to_numpy(runner.output[self.output_key].detach())
        targets = to_numpy(runner.input[self.input_key].detach())

        for img_id, y_true, y_pred in zip(image_ids, targets, outputs):
            if img_id not in self.scores_per_image:
                self.scores_per_image[img_id] = {"intersection": 0, "union": 0}

            y_true_labels = self.inputs_to_labels(y_true)
            y_pred_labels = self.outputs_to_labels(y_pred)
            intersection = (y_true_labels * y_pred_labels).sum()
            union = y_true_labels.sum() + y_pred_labels.sum() - intersection

            self.scores_per_image[img_id]["intersection"] += float(intersection)
            self.scores_per_image[img_id]["union"] += float(union)

    def on_loader_end(self, runner: IRunner):
        # Gather statistics from all nodes
        gathered_scores_per_image = all_gather(self.scores_per_image)
        all_scores_per_image = defaultdict(lambda: {"intersection": 0.0, "union": 0.0})
        for scores_per_image in gathered_scores_per_image:
            for image_id, values in scores_per_image.items():
                all_scores_per_image[image_id]["intersection"] += values["intersection"]
                all_scores_per_image[image_id]["union"] += values["union"]

        eps = 1e-7
        ious_per_image = []
        ious_per_location = defaultdict(list)

        for image_id, values in all_scores_per_image.items():
            intersection = values["intersection"]
            union = values["union"]
            metric = intersection / (union + eps)
            ious_per_image.append(metric)

            for location in self.locations:
                if str.startswith(image_id, location):
                    ious_per_location[location].append(metric)

        metric = float(np.mean(ious_per_image))
        runner.loader_metrics[self.prefix] = metric

        for location, ious in ious_per_location.items():
            runner.loader_metrics[f"{self.prefix}/{location}"] = float(np.mean(ious))


class JaccardMetricPerImageWithOptimalThreshold(Callback):
    """
    Callback that computes an optimal threshold for binarizing logits and theoretical IoU score at given threshold.
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        image_id_key: str = "image_id",
        prefix: str = "optimal_threshold",
    ):
        super().__init__(CallbackOrder.Metric)
        """
        :param input_key: input key to use for precision calculation; specifies our `y_true`.
        :param output_key: output key to use for precision calculation; specifies our `y_pred`.
        """
        self.prefix = prefix
        self.output_key = output_key
        self.input_key = input_key
        self.image_id_key = image_id_key
        self.thresholds = torch.arange(0.3, 0.6, 0.025).detach()
        self.scores_per_image = {}

    def on_loader_start(self, runner: IRunner):
        self.scores_per_image = {}

    @torch.no_grad()
    def on_batch_end(self, runner: IRunner):
        image_id = runner.input[self.image_id_key]
        outputs = runner.output[self.output_key].detach().sigmoid()
        targets = runner.input[self.input_key].detach()

        # Flatten images for easy computing IoU
        assert outputs.size(1) == 1
        assert targets.size(1) == 1
        outputs = outputs.view(outputs.size(0), -1, 1) > self.thresholds.to(outputs.dtype).to(outputs.device).view(
            1, 1, -1
        )
        targets = targets.view(targets.size(0), -1) == 1
        n = len(self.thresholds)

        for i, threshold in enumerate(self.thresholds):
            # Binarize outputs
            outputs_i = outputs[..., i]
            intersection = torch.sum(targets & outputs_i, dim=1)
            union = torch.sum(targets | outputs_i, dim=1)

            for img_id, img_intersection, img_union in zip(image_id, intersection, union):
                if img_id not in self.scores_per_image:
                    self.scores_per_image[img_id] = {"intersection": np.zeros(n), "union": np.zeros(n)}

                self.scores_per_image[img_id]["intersection"][i] += float(img_intersection)
                self.scores_per_image[img_id]["union"][i] += float(img_union)

    def on_loader_end(self, runner: IRunner):
        eps = 1e-7
        ious_per_image = []

        # Gather statistics from all nodes
        all_gathered_scores_per_image = all_gather(self.scores_per_image)

        n = len(self.thresholds)
        all_scores_per_image = defaultdict(lambda: {"intersection": np.zeros(n), "union": np.zeros(n)})
        for scores_per_image in all_gathered_scores_per_image:
            for image_id, values in scores_per_image.items():
                all_scores_per_image[image_id]["intersection"] += values["intersection"]
                all_scores_per_image[image_id]["union"] += values["union"]

        for image_id, values in all_scores_per_image.items():
            intersection = values["intersection"]
            union = values["union"]
            metric = intersection / (union + eps)
            ious_per_image.append(metric)

        thresholds = to_numpy(self.thresholds)
        iou = np.mean(ious_per_image, axis=0)
        assert len(iou) == len(thresholds)

        threshold_index = np.argmax(iou)
        iou_at_threshold = iou[threshold_index]
        threshold_value = thresholds[threshold_index]

        runner.loader_metrics[self.prefix + "/" + "threshold"] = float(threshold_value)
        runner.loader_metrics[self.prefix] = float(iou_at_threshold)

        if get_rank() in {-1, 0}:
            logger = get_tensorboard_logger(runner)
            logger.add_histogram(self.prefix, iou, global_step=runner.epoch)
