from collections import defaultdict
import numpy as np
import torch
from catalyst.dl.callbacks import Callback, RunnerState
from pytorch_toolbelt.utils.catalyst_utils import _get_tensorboard_logger


class JaccardMetricPerImage(Callback):
    """
    Jaccard metric callback which computes IoU metric per image and is aware that image is tiled.
    """

    def __init__(self, input_key: str = "targets", output_key: str = "logits", image_id_key: str = "image_id", prefix: str = "jaccard"):
        """
        :param input_key: input key to use for precision calculation; specifies our `y_true`.
        :param output_key: output key to use for precision calculation; specifies our `y_pred`.
        """
        self.prefix = prefix
        self.output_key = output_key
        self.input_key = input_key
        self.image_id_key = image_id_key
        self.scores_per_image = defaultdict(lambda: {'intersection': 0.0, 'union': 0.0})
        self.locations = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']

    def on_loader_start(self, state):
        self.scores_per_image = defaultdict(lambda: {'intersection': 0.0, 'union': 0.0})

    def on_batch_end(self, state: RunnerState):
        image_id = state.input[self.image_id_key]
        outputs = state.output[self.output_key].detach()
        targets = state.input[self.input_key].detach()

        # Flatten images for easy computing IoU
        outputs = outputs.view(outputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        # Binarize outputs as we don't want to compute soft-jaccard
        outputs = (outputs > 0).float()

        intersection = torch.sum(targets * outputs, dim=1)
        union = torch.sum(targets, dim=1) + torch.sum(outputs, dim=1)
        for img_id, img_intersection, img_union in zip(image_id, intersection, union):
            self.scores_per_image[img_id]['intersection'] += float(img_intersection)
            self.scores_per_image[img_id]['union'] += float(img_union)

    def on_loader_end(self, state):
        metric_name = self.prefix
        eps = 1e-7

        ious_per_image = []
        ious_per_location = defaultdict(list)

        for image_id, values in self.scores_per_image.items():
            intersection = values['intersection']
            union = values['union']
            metric = intersection / (union - intersection + eps)
            ious_per_image.append(metric)

            for location in self.locations:
                if str.startswith(image_id, location):
                    ious_per_location[location].append(metric)

        metric = float(np.mean(ious_per_image))
        state.metrics.epoch_values[state.loader_name][metric_name] = metric

        logger = _get_tensorboard_logger(state)
        logger.add_scalar(f"{self.prefix}/all", metric, global_step=state.epoch)

        for location, ious in ious_per_location.items():
            state.metrics.epoch_values[state.loader_name][f'{self.prefix}_{location}'] = float(np.mean(ious))
            logger.add_scalar(f"{self.prefix}/{location}", metric, global_step=state.epoch)
