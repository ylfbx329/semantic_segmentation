import argparse
import torch
from torch.utils.data import DataLoader
import models
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="pytorch fcn training")
    parser.add_argument("--data-path", default="./data/Pascal VOC 2012/VOCdevkit/VOC2012", help="VOC2012 root")
    parser.add_argument("--crop-size", default=(224, 224), type=tuple, help="image crop size")
    parser.add_argument("--num-classes", default=21, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=30, type=int, metavar="N", help="number of total epochs to train")
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.99, type=float, metavar='M', help='momentum')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size

    train_data = VOCDataset(root=args.data_path,
                            img_set="train",
                            crop_size=(224, 224))
    val_data = VOCDataset(root=args.data_path,
                          img_set="val",
                          crop_size=(224, 224))
    train_loader = DataLoader(dataset=train_data,
                              batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(dataset=val_data,
                            batch_size=batch_size,
                            shuffle=True)
    log_train_idx = len(train_loader) / 3
    log_val_idx = len(val_loader) / 3

    model = models.FCN()

    epochs = args.epochs
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = torch.nn.CrossEntropyLoss()

    # 运行主训练循环
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_train_idx == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_data), 100. * batch_idx / len(train_loader),
                    loss.data.item()))
