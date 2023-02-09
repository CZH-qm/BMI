# coding: utf-8
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import torch.nn as nn
import torch.optim as optim
import sys
from utils import MyDataset
from datetime import datetime
from tqdm import tqdm
from visdom import Visdom
import timm
from monai.losses import FocalLoss

torch.set_default_tensor_type(torch.DoubleTensor)

# train_txt_path = r'E:\my_code\bmi_code\newData\png\train.csv'
# valid_txt_path = r'E:\my_code\bmi_code\newData\png\valid.csv'
# test_path = r'E:\my_code\bmi_code\newData\png\test.csv'

train_txt_path = r'E:\my_code\bmi_code\newData\png\ce_train.csv'
valid_txt_path = r'E:\my_code\bmi_code\newData\png\ce_valid.csv'
test_path = r'E:\my_code\bmi_code\newData\png\ce_test.csv'

train_bs = 4  # train batch size
valid_bs = 4  # valid batch size
#lr_init = 0.01  # learning rate
lr_init = 0.001  # learning rate
max_epoch = 100  # 训练轮次
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
result_dir = "Result"
wind = Visdom()
wind.line([[0., 0.]], [0.], win='train', opts=dict(title='loss', legend=['train loss', 'val loss']))

now_time = datetime.now()  # 记录当前时间
time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
log_dir = os.path.join(result_dir, time_str)  # 日志的保存路径
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# ------------------------------------ step 1/5 : 加载数据------------------------------------
# 数据预处理设置
normMean = [0.7522175, 0.7522175, 0.7522175]
normStd = [0.20582062, 0.20582062, 0.20582062]
normTransform = transforms.Normalize(normMean, normStd)  # 归一化的公式
Transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    normTransform
])

# 构建MyDataset实例   把图片拿出来并且做预处理
train_data = MyDataset(txt_path=train_txt_path, transform=Transform)
valid_data = MyDataset(txt_path=valid_txt_path, transform=Transform)

# 构建DataLoder迭代器，把整体图片拿出来一部分放进迭代器里
train_loader = DataLoader(dataset=train_data, batch_size=train_bs, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=valid_bs)

# 创建分类网络
# net = timm.create_model('resnet18', pretrained=False, num_classes=2)
net = timm.create_model('convnext_small', pretrained=False, num_classes=2)
# net = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
# net = timm.create_model('vit_small_patch16_224', pretrained=True)


net.to(device)

# ------------------------------------ step 3/5 : 定义损失函数和优化器 ------------------------------------
# criterion = nn.CrossEntropyLoss()  # 选择损失函数
criterion = FocalLoss(to_onehot_y=True)

optimizer = optim.AdamW(net.parameters(), lr=lr_init)  # 选择优化器
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=64)

# ------------------------------------ step 4/5 : 训练 --------------------------------------------------
for epoch in range(max_epoch):
    loss_sigma = 0.0  # 记录一个epoch的loss之和
    net.train()  # 让网络进入训练状态
    for data in tqdm(train_loader):
        # 获取图片和标签
        inputs, labels = data  # data相当于一个列表，取出的是一个batch的输入和标签
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)  # 得到模型的输出
        loss = criterion(outputs, labels)

        optimizer.zero_grad()  # 相当于用损失去计算模型的更新参数
        loss.backward()
        optimizer.step()
        scheduler.step()  # 更新学习率
        loss_sigma += loss.item()
    loss_avg = loss_sigma / len(train_loader)

    # ------------------------------------ 观察模型在验证集上的表现 ------------------------------------
    loss_val = 0.0
    net.eval()  # 进入验证状态
    with torch.no_grad():
        for data in tqdm(valid_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss_val += loss.item()
        loss_val_avg = loss_val / len(valid_loader)
        print("Epoch{:0>3}/{:0>3} Train loss:{:.4f} Valid loss:{:.4f}".format(epoch + 1, max_epoch, loss_avg,
                                                                              loss_val_avg))
    wind.line([[loss_avg, loss_val_avg]], [epoch + 1], win='train', update='append')

    # ------------------------------------ step5: 保存模型 ------------------------------------
    if epoch % 10 == 9:
        net_save_path = os.path.join(log_dir, 'net_params_' + str(epoch + 1) + '.pkl')
        torch.save(net.state_dict(), net_save_path)
    print('Finished Training')
