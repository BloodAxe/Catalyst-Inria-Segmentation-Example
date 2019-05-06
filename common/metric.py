from collections import defaultdict

import torch
from catalyst.dl.callbacks import Callback, RunnerState
from pytorch_toolbelt.utils.catalyst_utils import _get_tensorboard_logger


class EpochJaccardMetric(Callback):
    """
    Jaccard metric callback which computes IoU metric per image and is aware that image is tiled.
    """

    def __init__(self, input_key: str = "targets", output_key: str = "logits", prefix: str = "jaccard"):
        """
        :param input_key: input key to use for precision calculation; specifies our `y_true`.
        :param output_key: output key to use for precision calculation; specifies our `y_pred`.
        """
        self.prefix = prefix
        self.output_key = output_key
        self.input_key = input_key
        self.scores_per_image = defaultdict()
        self.intersection = 0
        self.union = 0

    def on_loader_start(self, state):
        self.intersection = 0
        self.union = 0

    def on_batch_end(self, state: RunnerState):
        image_id = state.input['image_id']
        outputs = state.output[self.output_key].detach()
        targets = state.input[self.input_key].detach()

        # Binarize outputs as we don't want to compute soft-jaccard
        outputs = (outputs > 0).float()

        intersection = float(torch.sum(targets * outputs))
        union = float(torch.sum(targets) + torch.sum(outputs))
        self.intersection += intersection
        self.union += union

    def on_loader_end(self, state):
        metric_name = self.prefix
        eps = 1e-7
        metric = self.intersection / (self.union - self.intersection + eps)
        state.metrics.epoch_values[state.loader_name][metric_name] = metric

        logger = _get_tensorboard_logger(state)
        logger.add_scalar(f"{self.prefix}/epoch", metric, global_step=state.epoch)
