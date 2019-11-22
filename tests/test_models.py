import torch

from inria.dataset import OUTPUT_MASK_KEY
from inria.models.fpn import resnet34_rfpncat128


def test_resnet34_rfpncat128():
    net = resnet34_rfpncat128().cuda().eval()
    input = torch.rand((2, 3, 256, 256)).cuda()
    output = net(input)
    mask = output[OUTPUT_MASK_KEY]
    print(mask.mean(), mask.std())