# coding: utf-8
from PIL import Image
from torch.utils.data import Dataset
import torch
import pandas as pd

def age_class(age):
    if age<60:
        age=0
    if age<65 and age>=60:
        age=1
    if age<70 and age>=65:
        age=2
    if age<75 and age>=70:
        age=3
    return age
class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        data = pd.read_csv(txt_path)
        imgs, age, gender, bmi = data['root'], data['age'], data['gender'], data['bmi']

        self.imgs = imgs
        self.age = age
        self.gender = gender
        self.bmi = bmi

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, age, gender, bmi = self.imgs[index], self.age[index], self.gender[index], self.bmi[index]
        img = Image.open(fn).convert('RGB')  # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1
        age=age_class(age)
        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(age, dtype=torch.int64), torch.tensor(gender, dtype=torch.int64), \
               torch.tensor(bmi, dtype=torch.float64)

    def __len__(self):
        return len(self.imgs)
