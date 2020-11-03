import pytest
import torch
from pytorch_toolbelt.utils.torch_utils import maybe_cuda, count_parameters

from inria.models import get_model
from inria.models.efficient_unet import b4_effunet32_s2
from inria.models.u2net import U2NETP, U2NET
from inria.models.unet import b6_unet32_s2_rdtc, b6_unet32_s2_tc


@torch.no_grad()
def test_b4_effunet32_s2():
    model = maybe_cuda(b4_effunet32_s2())
    x = maybe_cuda(torch.rand((2, 3, 512, 512)))
    output = model(x)
    print(count_parameters(model))
    for key, value in output.items():
        print(key, value.size(), value.mean(), value.std())


@torch.no_grad()
def test_b6_unet32_s2_tc():
    model = b4_effunet32_s2()
    model = maybe_cuda(model.eval())
    x = maybe_cuda(torch.rand((2, 3, 512, 512)))
    output = model(x)
    print(count_parameters(model))
    for key, value in output.items():
        print(key, value.size(), value.mean(), value.std())


@torch.no_grad()
def test_b6_unet32_s2_rdtc():
    model = b6_unet32_s2_rdtc(need_supervision_masks=True)
    model = maybe_cuda(model.eval())
    x = maybe_cuda(torch.rand((2, 3, 512, 512)))
    output = model(x)
    print(count_parameters(model))
    for key, value in output.items():
        print(key, value.size(), value.mean(), value.std())


@torch.no_grad()
def test_test_b6_unet32_s2_tc():
    model = b6_unet32_s2_tc()
    model = maybe_cuda(model.eval())
    x = maybe_cuda(torch.rand((2, 3, 512, 512)))
    output = model(x)
    print(count_parameters(model))
    for key, value in output.items():
        print(key, value.size(), value.mean(), value.std())


@torch.no_grad()
def test_U2NETP():
    model = U2NETP()
    model = maybe_cuda(model.eval())
    x = maybe_cuda(torch.rand((2, 3, 512, 512)))
    output = model(x)
    print(count_parameters(model))
    for key, value in output.items():
        print(key, value.size(), value.mean(), value.std())


@torch.no_grad()
def test_U2NET():
    model = U2NET()
    model = maybe_cuda(model.eval())
    x = maybe_cuda(torch.rand((2, 3, 512, 512)))
    output = model(x)
    print(count_parameters(model))
    for key, value in output.items():
        print(key, value.size(), value.mean(), value.std())
