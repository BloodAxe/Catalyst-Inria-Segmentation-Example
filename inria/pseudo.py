import numpy as np
from catalyst.dl import Callback, CallbackOrder, IRunner
from pytorch_toolbelt.utils.catalyst import PseudolabelDatasetMixin
from pytorch_toolbelt.utils.torch_utils import to_numpy

__all__ = ["BCEOnlinePseudolabelingCallback2d"]


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
        pseudolabel_loader="infer",
        prob_threshold=0.9,
        sample_index_key="index",
        output_key="logits",
        unlabeled_class=-100,
        label_smoothing=0.0,
        label_frequency=1,
    ):
        assert 1.0 > prob_threshold > 0.5

        super().__init__(CallbackOrder.External)
        self.unlabeled_ds = unlabeled_ds
        self.pseudolabel_loader = pseudolabel_loader
        self.prob_threshold = prob_threshold
        self.sample_index_key = sample_index_key
        self.output_key = output_key
        self.unlabeled_class = unlabeled_class
        self.label_smoothing = label_smoothing
        self.last_labeled_epoch = None
        self.label_frequency = label_frequency

    # def on_epoch_start(self, state: RunnerState):
    #     pass

    # def on_loader_start(self, state: RunnerState):
    #     if state.loader_name == self.pseudolabel_loader:
    #         self.predictions = []

    def on_stage_start(self, runner: IRunner):
        self.last_labeled_epoch = None

    def on_loader_start(self, runner: IRunner):
        if runner.loader_name == self.pseudolabel_loader:
            self.should_relabel = self.last_labeled_epoch is None or (
                runner.epoch == self.last_labeled_epoch + self.label_frequency
            )
            print("Should relabel", self.should_relabel, runner.epoch)

    def on_loader_end(self, runner: "IRunner"):
        if runner.loader_name == self.pseudolabel_loader and self.should_relabel:
            self.last_labeled_epoch = runner.epoch
            print("Set last_labeled_epoch", runner.epoch)

    def get_probabilities(self, state: IRunner):
        probs = state.output[self.output_key].detach().sigmoid()
        indexes = state.input[self.sample_index_key]

        return to_numpy(probs), to_numpy(indexes)

    def on_batch_end(self, runner: IRunner):
        if runner.loader_name != self.pseudolabel_loader:
            return

        if not self.should_relabel:
            return

        # Get predictions for batch
        probs, indexes = self.get_probabilities(runner)

        for p, sample_index in zip(probs, indexes):
            # confident_negatives = p < (1.0 - self.prob_threshold)
            # confident_positives = p > self.prob_threshold
            # rest = ~confident_negatives & ~confident_positives
            #
            # p = p.copy()
            # p[confident_negatives] = 0 + self.label_smoothing
            # p[confident_positives] = 1 - self.label_smoothing
            # p[rest] = self.unlabeled_class
            p = np.moveaxis(p, 0, -1)

            self.unlabeled_ds.set_target(sample_index, p)
