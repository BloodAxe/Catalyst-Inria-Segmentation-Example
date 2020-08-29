import pytest
import torch
from pytorch_toolbelt.utils.torch_utils import maybe_cuda, count_parameters

from inria.models import get_model
from inria.models.efficient_unet import b4_effunet32_s2


@torch.no_grad()
def test_b4_effunet32_s2():
    model = maybe_cuda(b4_effunet32_s2())
    x = maybe_cuda(torch.rand((2, 3, 512, 512)))
    output = model(x)
    print(count_parameters(model))
    for key, value in output.items():
        print(key, value.size(), value.mean(), value.std())


