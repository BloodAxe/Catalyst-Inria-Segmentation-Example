import pytest
import torch
from pytorch_toolbelt.utils.torch_utils import maybe_cuda, count_parameters

from inria.models import get_model, MODEL_REGISTRY


@torch.no_grad()
@pytest.mark.parametrize("model_name", MODEL_REGISTRY.keys())
def test_models(model_name):
    model = maybe_cuda(get_model(model_name, pretrained=False).eval())
    x = maybe_cuda(torch.rand((2, 3, 256, 256)))
    output = model(x)
    print(model_name, count_parameters(model))
    for key, value in output.item():
        print(key, value.size(), value.mean(), value.std())
