# coding: utf-8
from PIL import Image
from torch.utils.data import Dataset
import torch
import pandas as pd


class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        data = pd.read_csv(txt_path)
        imgs, labels = data['root'], data['gender']  # 这里需要加上年龄、性别的标签

        self.imgs = imgs  # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index], self.labels[index]
        img = Image.open(fn).convert('RGB')  # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1

        if self.transform is not None:
            img = self.transform(img)  # 在这里做transform，转为tensor等等

        return img, torch.tensor(label, dtype=torch.int64)

    def __len__(self):
        return len(self.imgs)
