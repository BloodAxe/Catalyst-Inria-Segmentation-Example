from catalyst.dl import Callback, CallbackOrder, RunnerState
from pytorch_toolbelt.utils.catalyst import PseudolabelDatasetMixin
from pytorch_toolbelt.utils.torch_utils import to_numpy
import numpy as np

class BCEOnlinePseudolabelingCallback2d(Callback):
    """
    Online pseudo-labeling callback for multi-class problem.

    >>> unlabeled_train = get_test_dataset(
    >>>     data_dir, image_size=image_size, augmentation=augmentations
    >>> )
    >>> unlabeled_eval = get_test_dataset(
    >>>     data_dir, image_size=image_size
    >>> )
    >>>
    >>> callbacks += [
    >>>     MulticlassOnlinePseudolabelingCallback(
    >>>         unlabeled_train.targets,
    >>>         pseudolabel_loader="label",
    >>>         prob_threshold=0.9)
    >>> ]
    >>> train_ds = train_ds + unlabeled_train
    >>>
    >>> loaders = collections.OrderedDict()
    >>> loaders["train"] = DataLoader(train_ds)
    >>> loaders["valid"] = DataLoader(valid_ds)
    >>> loaders["label"] = DataLoader(unlabeled_eval, shuffle=False) # ! shuffle=False is important !
    """

    def __init__(
        self,
        unlabeled_ds: PseudolabelDatasetMixin,
        pseudolabel_loader="label",
        prob_threshold=0.9,
        sample_index_key="index",
        output_key="logits",
        unlabeled_class=-100,
        label_smoothing=0.0,
    ):
        assert 1.0 > prob_threshold > 0.5

        super().__init__(CallbackOrder.Other)
        self.unlabeled_ds = unlabeled_ds
        self.pseudolabel_loader = pseudolabel_loader
        self.prob_threshold = prob_threshold
        self.sample_index_key = sample_index_key
        self.output_key = output_key
        self.unlabeled_class = unlabeled_class
        self.label_smoothing = label_smoothing

    # def on_epoch_start(self, state: RunnerState):
    #     pass

    # def on_loader_start(self, state: RunnerState):
    #     if state.loader_name == self.pseudolabel_loader:
    #         self.predictions = []

    def get_probabilities(self, state: RunnerState):
        probs = state.output[self.output_key].detach().sigmoid()
        indexes = state.input[self.sample_index_key]

        return to_numpy(probs), to_numpy(indexes)

    def on_batch_end(self, state: RunnerState):
        if state.loader_name != self.pseudolabel_loader:
            return

        # Get predictions for batch
        probs, indexes = self.get_probabilities(state)

        for p, sample_index in zip(probs, indexes):
            confident_negatives = p < (1.0 - self.prob_threshold)
            confident_positives = p > self.prob_threshold
            rest = ~confident_negatives & ~confident_positives

            p = p.copy()
            p[confident_negatives] = 0 + self.label_smoothing
            p[confident_positives] = 1 - self.label_smoothing
            p[rest] = self.unlabeled_class
            p = np.moveaxis(p, 0, -1)

            self.unlabeled_ds.set_target(sample_index, p)

    # def on_loader_end(self, state: RunnerState):
    # if state.loader_name != self.pseudolabel_loader:
    #     return

    # predictions = np.array(self.predictions)
    # max_pred = np.argmax(predictions, axis=1)
    # max_score = np.amax(predictions, axis=1)
    # confident_mask = max_score > self.prob_threshold
    # num_samples = len(predictions)
    #
    # for index, predicted_target, score in zip(range(num_samples, max_pred, max_score)):
    #     target = predicted_target if score > self.prob_threshold else self.unlabeled_class
    #     self.unlabeled_ds.set_target(index, target)
    #
    # num_confident_samples = confident_mask.sum()
    # state.metrics.epoch_values[state.loader_name]["pseudolabeling/confident_samples"] = num_confident_samples
    # state.metrics.epoch_values[state.loader_name]["pseudolabeling/confident_samples_mean_score"] = max_score[
    #     confident_mask
    # ].mean()
    #
    # state.metrics.epoch_values[state.loader_name]["pseudolabeling/unconfident_samples"] = (
    #     len(predictions) - num_confident_samples
    # )
    # state.metrics.epoch_values[state.loader_name]["pseudolabeling/unconfident_samples_mean_score"] = max_score[
    #     ~confident_mask
    # ].mean()

    # def on_epoch_end(self, state: RunnerState):
    #     pass
