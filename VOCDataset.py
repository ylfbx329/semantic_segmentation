import json
import os.path
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co
from torchvision import transforms
from torchvision.io import read_image
import torchvision.transforms.functional as ttf


class VOCDataset(Dataset):
    def __init__(self, root, img_set, crop_size):
        super().__init__()
        # root = r"./data/Pascal VOC 2012/VOCdevkit/VOC2012"
        self.root = root
        self.img_set = img_set
        self.crop_size = crop_size
        self.txt_path = str(Path(self.root) / ("ImageSets/Segmentation/" + self.img_set + ".txt"))
        with open(self.txt_path, "r", encoding="utf-8") as f:
            self.filename_list = f.read().splitlines()
        random.shuffle(self.filename_list)
        self.image_list = [read_image(str(Path(root) / "JPEGImages" / (str(filename) + ".jpg")))
                           for filename in self.filename_list]
        self.target_list = [read_image(str(Path(root) / "SegmentationClass" / (str(filename) + ".png")))
                            for filename in self.filename_list]
        self.image_list = [img
                           for img in self.image_list
                           if img.shape[1] >= crop_size[0] and img.shape[2] >= crop_size[1]]
        self.target_list = [img.where(img != 255, 0)
                            for img in self.target_list
                            if img.shape[1] >= crop_size[0] and img.shape[2] >= crop_size[1]]
        self.length = len(self.image_list)
        self.color_json_path = str(Path(root) / "ImageSets/Segmentation/palette.json")
        with open(self.color_json_path, "r", encoding="utf-8") as f:
            self.color_dict = json.load(f)
        self.class_json_path = str(Path(root) / "ImageSets/Segmentation/pascal_voc_classes.json")
        with open(self.class_json_path, "r", encoding="utf-8") as f:
            self.class_dict = json.load(f)

    def __getitem__(self, index) -> T_co:
        image = self.image_list[index]
        target = self.target_list[index]
        position = transforms.RandomCrop.get_params(image, self.crop_size)
        image = ttf.crop(image, *position).type(torch.float)
        target = ttf.crop(target, *position).type(torch.int64)
        target = target.squeeze(0)
        return image, target

    def __add__(self, other: 'Dataset[T_co]') -> 'ConcatDataset[T_co]':
        return super().__add__(other)

    def __len__(self) -> int:
        return self.length


if __name__ == "__main__":
    batch_size = 100
    train_data = VOCDataset(root="./data/Pascal VOC 2012/VOCdevkit/VOC2012", img_set="train", crop_size=(224, 224))
    val_data = VOCDataset(root="./data/Pascal VOC 2012/VOCdevkit/VOC2012", img_set="val", crop_size=(224, 224))
    train_loader = DataLoader(dataset=train_data,
                              batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(dataset=val_data,
                            batch_size=batch_size,
                            shuffle=True)
    for batch_idx, (data, target) in enumerate(train_loader):
        print(data.dtype)
    # 运行主训练循环
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            # 将数据大小从 (batch_size, 1, 28, 28) 变为 (batch_size, 28*28)
            # data = data.view(-1, 28 * 28)
            output = model(data)
            # normalized_masks = output.softmax(dim=1)
            # print(normalized_masks.size())
            # out_masks = normalized_masks.argmax(1) == torch.arange(num_classes)[:, None, None, None]
            # out_masks = out_masks.swapaxes(0, 1)
            # target_mask = target == torch.arange(num_classes)[:, None, None, None]
            # print(output, target.type(torch.int32))
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 1000 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data.item()))
