import random
from pathlib import Path
import torch
from torchvision import transforms
from torchvision.io import read_image
import torchvision.transforms.functional as F
from torch.utils.data.dataset import T_co, Dataset


class VOCSegmentation(Dataset):
    r"""
    Pascal VOC 2012语义分割数据集

    Args:
        root: Pascal VOC 2012数据集根目录
        img_set: 选择训练集或验证集
        crop_size: 随机裁剪尺寸
        pad: 四周镜像填充像素数
        shuffle: 是否打乱数据集

    TODO 验证可行性，提高扩展性
    """

    def __init__(self,
                 root=r"./data/Pascal VOC 2012/VOCdevkit/VOC2012",
                 img_set="train",
                 crop_size=224,
                 pad=None,
                 shuffle=True):
        super().__init__()
        self.root = root
        self.img_set = img_set
        self.crop_size = crop_size if isinstance(crop_size, tuple) else (crop_size, crop_size)
        self.pad = pad if isinstance(pad, tuple) else (pad, pad, pad, pad)
        self.txt_path = str(Path(self.root) / ("ImageSets/Segmentation/" + self.img_set + ".txt"))
        with open(self.txt_path, "r", encoding="utf-8") as f:
            self.filename_list = f.read().splitlines()
        if shuffle is True:
            random.shuffle(self.filename_list)
        # 每个图像torch.Size([c, x, y])
        self.image_list = [read_image(str(Path(root) / "JPEGImages" / (str(filename) + ".jpg")))
                           for filename in self.filename_list]
        self.target_list = [read_image(str(Path(root) / "SegmentationClass" / (str(filename) + ".png")))
                            for filename in self.filename_list]
        self.image_list = self.filter(self.image_list)
        self.target_list = self.filter(self.target_list)
        self.target_list = self.tensor2label()
        self.length = len(self.image_list)

    def filter(self, image_list):
        # 筛选尺寸大于等于随机裁剪尺寸的图片
        return [img
                for img in image_list
                if img.shape[1] >= self.crop_size[0] and img.shape[2] >= self.crop_size[1]]

    def tensor2label(self):
        # 删除255类别，使包括背景共有21种类别
        return [img.where(img != 255, 0)[0] for img in self.target_list]

    def transform(self, image, target):
        position = transforms.RandomCrop.get_params(image, self.crop_size)
        # 参数加星号表示解包，以适应不同参数个数定义
        image = F.crop(image, *position).type(torch.float)
        target = F.crop(target, *position).type(torch.long)
        if self.pad is not None:
            image = torch.nn.functional.pad(input=image, pad=self.pad, mode="reflect")
        return image, target

    def __getitem__(self, index) -> T_co:
        return self.transform(self.image_list[index], self.target_list[index])

    def __len__(self) -> int:
        return self.length
