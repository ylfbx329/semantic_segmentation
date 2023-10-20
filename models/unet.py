from collections import OrderedDict

import torch
import torchvision
from torch import nn


class UNet(nn.Module):
    """
    自己实现U-Net

    Args:
        in_channels: 输入通道数
        num_classes: 类别数

    TODO 调整网络结构，验证模型可行性
    """

    def __init__(self, in_channels=1, num_classes=21):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.transform = torchvision.transforms.CenterCrop(size=0)
        self.layer1 = nn.Sequential(OrderedDict([
            ("conv3-in-64", nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=3)),
            ("relu", nn.ReLU(inplace=True)),
            ("conv3-64", nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)),
            ("relu", nn.ReLU(inplace=True))
        ]))
        self.layer2 = nn.Sequential(OrderedDict([
            ("pool2", nn.MaxPool2d(kernel_size=2)),
            ("conv3-64-128", nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)),
            ("relu", nn.ReLU(inplace=True)),
            ("conv3-128", nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)),
            ("relu", nn.ReLU(inplace=True))
        ]))
        self.layer3 = nn.Sequential(OrderedDict([
            ("pool2", nn.MaxPool2d(kernel_size=2)),
            ("conv3-128-256", nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)),
            ("relu", nn.ReLU(inplace=True)),
            ("conv3-256", nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)),
            ("relu", nn.ReLU(inplace=True))
        ]))
        self.layer4 = nn.Sequential(OrderedDict([
            ("pool2", nn.MaxPool2d(kernel_size=2)),
            ("conv3-256-512", nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)),
            ("relu", nn.ReLU(inplace=True)),
            ("conv3-512", nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)),
            ("relu", nn.ReLU(inplace=True))
        ]))
        self.layer5 = nn.Sequential(OrderedDict([
            ("pool2", nn.MaxPool2d(kernel_size=2)),
            ("conv3-512-1024", nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3)),
            ("relu", nn.ReLU(inplace=True)),
            ("conv3-1024", nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3)),
            ("relu", nn.ReLU(inplace=True)),
            # 2H = SH - S - 2P + K
            # S = 2, K = S + 2P
            ("convT2s2-1024-512",
             nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2, padding=0))
        ]))
        self.layer6 = nn.Sequential(OrderedDict([
            ("conv3-1024-512", nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3)),
            ("relu", nn.ReLU(inplace=True)),
            ("conv3-512", nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)),
            ("relu", nn.ReLU(inplace=True)),
            ("convT2s2-512-256",
             nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=0))
        ]))
        self.layer7 = nn.Sequential(OrderedDict([
            ("conv3-512-256", nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3)),
            ("relu", nn.ReLU(inplace=True)),
            ("conv3-256", nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)),
            ("relu", nn.ReLU(inplace=True)),
            ("convT2s2-256-128",
             nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0))
        ]))
        self.layer8 = nn.Sequential(OrderedDict([
            ("conv3-256-128", nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3)),
            ("relu", nn.ReLU(inplace=True)),
            ("conv3-128", nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)),
            ("relu", nn.ReLU(inplace=True)),
            ("convT2s2-128-64",
             nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0))
        ]))
        self.layer9 = nn.Sequential(OrderedDict([
            ("conv3-128-64", nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3)),
            ("relu", nn.ReLU(inplace=True)),
            ("conv3-64", nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)),
            ("relu", nn.ReLU(inplace=True)),
            ("classifier", nn.Conv2d(in_channels=64, out_channels=self.num_classes, kernel_size=1))
        ]))

    def forward(self, x):
        x = self.layer1(x)
        out1 = x.clone().detach().requires_grad_(True)
        x = self.layer2(x)
        out2 = x.clone().detach().requires_grad_(True)
        x = self.layer3(x)
        out3 = x.clone().detach().requires_grad_(True)
        x = self.layer4(x)
        out4 = x.clone().detach().requires_grad_(True)
        x = self.layer5(x)
        self.transform.size = x.shape[-2:]
        out4 = self.transform(out4)
        x = torch.cat([x, out4], dim=1)
        x = self.layer6(x)
        self.transform.size = x.shape[-2:]
        out3 = self.transform(out3)
        x = torch.cat([x, out3], dim=1)
        x = self.layer7(x)
        self.transform.size = x.shape[-2:]
        out2 = self.transform(out2)
        x = torch.cat([x, out2], dim=1)
        x = self.layer8(x)
        self.transform.size = x.shape[-2:]
        out1 = self.transform(out1)
        x = torch.cat([x, out1], dim=1)
        x = self.layer9(x)
        return x
