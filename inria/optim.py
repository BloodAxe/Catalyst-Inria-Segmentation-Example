from catalyst.contrib.optimizers import RAdam, Lamb
from torch.optim import SGD, Adam, RMSprop, AdamW


def get_optimizer(optimizer_name: str, parameters, learning_rate: float, weight_decay=1e-5, **kwargs):
    if optimizer_name.lower() == "sgd":
        return SGD(parameters, learning_rate, momentum=0.9, nesterov=True, weight_decay=weight_decay, **kwargs)

    if optimizer_name.lower() == "adam":
        return Adam(parameters, learning_rate, weight_decay=weight_decay, eps=1e-5, **kwargs)  # As Jeremy suggests

    if optimizer_name.lower() == "rms":
        return RMSprop(parameters, learning_rate, weight_decay=weight_decay, **kwargs)

    if optimizer_name.lower() == "adamw":
        return AdamW(parameters, learning_rate, weight_decay=weight_decay, eps=1e-5, **kwargs)

    if optimizer_name.lower() == "radam":
        return RAdam(parameters, learning_rate, weight_decay=weight_decay, eps=1e-5, **kwargs)  # As Jeremy suggests

    # if optimizer_name.lower() == "ranger":
    #     return Ranger(parameters, learning_rate, weight_decay=weight_decay,
    #                   **kwargs)

    # if optimizer_name.lower() == "qhadamw":
    #     return QHAdamW(parameters, learning_rate, weight_decay=weight_decay,
    #                    **kwargs)
    #
    if optimizer_name.lower() == "lamb":
        return Lamb(parameters, learning_rate, weight_decay=weight_decay, **kwargs)

    if optimizer_name.lower() == "fused_lamb":
        from apex.optimizers import FusedLAMB

        return FusedLAMB(parameters, learning_rate, weight_decay=weight_decay, **kwargs)

    if optimizer_name.lower() == "fused_adam":
        from apex.optimizers import FusedAdam

        return FusedAdam(parameters, learning_rate, eps=1e-5, weight_decay=weight_decay, adam_w_mode=True, **kwargs)

    raise ValueError("Unsupported optimizer name " + optimizer_name)
