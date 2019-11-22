import torch

__all__ = ["report_checkpoint", "clean_checkpoint"]


def report_checkpoint(checkpoint):
    print("Epoch          :", checkpoint["epoch"])
    print("Metrics (Train):", checkpoint["epoch_metrics"]["train"])
    print("Metrics (Valid):", checkpoint["epoch_metrics"]["valid"])


def clean_checkpoint(src_fname, dst_fname):
    checkpoint = torch.load(src_fname)

    keys = ["criterion_state_dict", "optimizer_state_dict", "scheduler_state_dict"]

    for key in keys:
        if key in checkpoint:
            del checkpoint[key]

    torch.save(checkpoint, dst_fname)
