from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.io import read_image
import torchvision.transforms.functional as F
import json
import random

import numpy as np
import torchvision.transforms
from matplotlib import pyplot as plt
from pathlib import Path
import torch
import torchvision.transforms.functional as ttf
from torchvision.io import read_image
from torch.nn import functional as nnf
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import optim
from pathlib import Path
import random

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co
from torchvision import transforms
from torchvision.io import read_image


class MyFCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=100)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(kernel_size=2, stride=2))
        ]))
        self.layer2 = nn.Sequential(OrderedDict([
            ('conv3', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)),
            ('relu3', nn.ReLU(inplace=True)),
            ('conv4', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)),
            ('relu4', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(kernel_size=2, stride=2))
        ]))
        self.layer3 = nn.Sequential(OrderedDict([
            ('conv5', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)),
            ('relu5', nn.ReLU(inplace=True)),
            ('conv6', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)),
            ('relu6', nn.ReLU(inplace=True)),
            ('conv7', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)),
            ('relu7', nn.ReLU(inplace=True)),
            ('pool3', nn.MaxPool2d(kernel_size=2, stride=2))
        ]))
        self.layer4 = nn.Sequential(OrderedDict([
            ('conv8', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)),
            ('relu8', nn.ReLU(inplace=True)),
            ('conv9', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)),
            ('relu9', nn.ReLU(inplace=True)),
            ('conv10', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)),
            ('relu10', nn.ReLU(inplace=True)),
            ('pool4', nn.MaxPool2d(kernel_size=2, stride=2))
        ]))
        self.layer5 = nn.Sequential(OrderedDict([
            ('conv11', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)),
            ('relu11', nn.ReLU(inplace=True)),
            ('conv12', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)),
            ('relu12', nn.ReLU(inplace=True)),
            ('conv13', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)),
            ('relu13', nn.ReLU(inplace=True)),
            ('pool5', nn.MaxPool2d(kernel_size=2, stride=2))
        ]))
        self.fc6 = nn.Sequential(OrderedDict([
            ('conv14', nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=7, padding=0)),
            ('relu14', nn.ReLU(inplace=True)),
            ('dropout1', nn.Dropout(p=0.5, inplace=True))
        ]))
        self.fc7 = nn.Sequential(OrderedDict([
            ('conv15', nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1, padding=0)),
            ('relu15', nn.ReLU(inplace=True)),
            ('dropout2', nn.Dropout(p=0.5, inplace=True))
        ]))
        self.classifier = nn.Conv2d(in_channels=4096, out_channels=21, kernel_size=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.zeros_(m.weight)

    def forward(self, x):
        # 记录输入图像大小
        input_shape = x.shape[-2:]
        # print(x.size())
        x = self.layer1(x)
        # print(x.size())
        x = self.layer2(x)
        # print(x.size())
        x = self.layer3(x)
        # print(x.size())
        x = self.layer4(x)
        # print(x.size())
        x = self.layer5(x)
        # print(x.size())
        x = self.fc6(x)
        # print(x.size())
        x = self.fc7(x)
        # print(x.size())
        x = self.classifier(x)
        # print(x.size())
        # 换变量名，否则指向相同空间会报错
        out = x.clone().detach().requires_grad_(True)
        out = nnf.interpolate(out, size=input_shape, mode="bilinear", align_corners=False)
        # print(x.size())
        return out


if __name__ == "__main__":
    model = MyFCN()
    input = read_image("./data/Pascal VOC 2012/VOCdevkit/VOC2012/JPEGImages/2007_000027.jpg")
    # transform = transforms.RandomCrop(224)
    # input = transform(input)
    input = transforms.functional.convert_image_dtype(input, torch.float)

    model.eval()
    output = model(input.unsqueeze(0))
    # print(output.size())
