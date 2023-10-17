import random
from pathlib import Path
import torch
from torchvision import transforms
from torchvision.io import read_image
import torchvision.transforms.functional as F
from torch.utils.data.dataset import T_co, Dataset


class VOCDataset(Dataset):
    r"""
    Pascal VOC 2012数据集

    Args:
        root: Pascal VOC 2012数据集根目录
        img_set: 选择训练集或验证集
        crop_size: 随机裁剪尺寸
        shuffle: 是否打乱数据集

    TODO 添加transform参数控制自定义变化
    """

    def __init__(self,
                 root=r"./data/Pascal VOC 2012/VOCdevkit/VOC2012",
                 img_set="train",
                 crop_size=(224, 224),
                 shuffle=True):
        super().__init__()
        self.root = root
        self.img_set = img_set
        self.crop_size = crop_size
        self.txt_path = str(Path(self.root) / ("ImageSets/Segmentation/" + self.img_set + ".txt"))
        with open(self.txt_path, "r", encoding="utf-8") as f:
            self.filename_list = f.read().splitlines()
        if shuffle:
            random.shuffle(self.filename_list)
        self.image_list = [read_image(str(Path(root) / "JPEGImages" / (str(filename) + ".jpg")))
                           for filename in self.filename_list]
        self.target_list = [read_image(str(Path(root) / "SegmentationClass" / (str(filename) + ".png")))
                            for filename in self.filename_list]
        # torch.Size([1, x, y])
        # 筛选尺寸大于等于随机裁剪尺寸的图片
        self.image_list = [img
                           for img in self.image_list
                           if img.shape[1] >= crop_size[0] and img.shape[2] >= crop_size[1]]
        # 删除255类别，使包括背景共有21种类别
        self.target_list = [img.where(img != 255, 0)
                            for img in self.target_list
                            if img.shape[1] >= crop_size[0] and img.shape[2] >= crop_size[1]]
        self.length = len(self.image_list)

    def __getitem__(self, index) -> T_co:
        image = self.image_list[index]
        target = self.target_list[index]
        position = transforms.RandomCrop.get_params(image, self.crop_size)
        image = F.crop(image, *position).type(torch.float)
        target = F.crop(target, *position).type(torch.long)
        return image, target[0]

    def __len__(self) -> int:
        return self.length
