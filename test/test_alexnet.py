import oneflow as flow
import oneflow.nn as nn
import numpy as np
import unittest
import sys
sys.path.append(r'../one-fx')
from onefx import symbolic_trace
import onefx
from oneflow.test_utils.automated_test_util import *


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: flow.Tensor) -> flow.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = flow.flatten(x, 1)
        x = self.classifier(x)
        return x


class TestAlexNet(unittest.TestCase):
    def test_alexnet(test_case):
        m = AlexNet()
        m = m.eval()
        gm: onefx.GraphModule = symbolic_trace(m)
        input = flow.randn(1, 3, 224, 224)
        assert np.allclose(gm(input).numpy(), m(input).numpy(), equal_nan=True)


if __name__ == "__main__":
    unittest.main()