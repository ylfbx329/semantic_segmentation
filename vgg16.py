import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.io import read_image


class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.batchNorm = nn.BatchNorm2d(64)  # pytorch的vgg16没有batchNorm
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(7 * 7 * 512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax()
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.zeros_(m.weight)

    def forward(self, x):
        print(x.size())
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool1(x)
        print(x.size())

        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool1(x)
        print(x.size())

        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv6(x))
        x = self.pool1(x)
        print(x.size())

        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        x = self.relu(self.conv8(x))
        x = self.pool1(x)
        print(x.size())

        x = self.relu(self.conv8(x))
        x = self.relu(self.conv8(x))
        x = self.relu(self.conv8(x))
        x = self.pool1(x)
        print(x.size())

        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        print(x.size())

        x = self.relu(self.fc2(x))
        x = self.dropout(x)

        x = self.fc3(x)
        print(x.size())
        return x


model = Module()
