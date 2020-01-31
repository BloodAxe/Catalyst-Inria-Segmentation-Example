import torch

from inria.dataset import OUTPUT_MASK_KEY
from inria.models.unet import hrnet18_unet64
from inria.models.runet import hrnet18_runet64
from pytorch_toolbelt.utils.torch_utils import count_parameters


@torch.no_grad()
def test_resnet34_rfpncat128():
    net = resnet34_rfpncat128().cuda().eval()
    input = torch.rand((2, 3, 256, 256)).cuda()
    output = net(input)
    mask = output[OUTPUT_MASK_KEY]
    print(mask.mean(), mask.std())



@torch.no_grad()
def test_hrnet18_runet64():
    net = hrnet18_runet64().eval()
    print(net)
    print(count_parameters(net))
    input = torch.rand((2, 3, 256, 256))
    output = net(input)
    mask = output[OUTPUT_MASK_KEY]
