import torch
import torch.nn as nn
from collections import OrderedDict


class VGG(nn.Module):
    """
    自己实现VGG16，删除最后softmax层，便于使用交叉熵损失

    Args:
        in_channels: 输入通道数
        num_classes: 类别数

    TODO 调整网络结构，验证模型可行性
    """

    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.layer1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=3, padding=100)),
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
        self.classifier = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(512 * 7 * 7, 4096)),
            ('relu14', nn.ReLU(inplace=True)),
            ('dropout1', nn.Dropout(p=0.5)),
            ('linear2', nn.Linear(4096, 4096)),
            ('relu15', nn.ReLU(inplace=True)),
            ('dropout2', nn.Dropout(p=0.5)),
            ('linear3', nn.Linear(4096, self.num_classes))
        ]))
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
