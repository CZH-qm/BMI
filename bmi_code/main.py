# coding: utf-8
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from model import Net

sys.path.append("..")
from utils import MyDataset
from datetime import datetime
from sklearn.metrics import mean_squared_error  # 均方误差
from sklearn.metrics import mean_absolute_error  # 平方绝对误差
from sklearn.metrics import r2_score  # R square
from tqdm import tqdm
from visdom import Visdom

torch.set_default_tensor_type(torch.DoubleTensor)


train_txt_path = r'C:\Users\Administrator\Desktop\diannao\bmi\newData\png\train.csv'
valid_txt_path = r'C:\Users\Administrator\Desktop\diannao\bmi\newData\png\valid.csv'
test_txt_path = r'C:\Users\Administrator\Desktop\diannao\bmi\newData\png\test.csv'

train_bs = 4   # batch size
valid_bs = 4
test_bs = 4
lr_init = 0.001
max_epoch = 100  # 训练轮次
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
# log  日志文件夹
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

trainTransform = transforms.Compose([  # 图形变换的处理流程
    transforms.Resize([224, 224]),  # padding or crop 原图像padding 1：1   monai croporpadding
    transforms.ToTensor(),
    normTransform
])

validTransform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    normTransform
])

testTransform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    normTransform
])

# 构建MyDataset实例   把图片拿出来并且做了处理
train_data = MyDataset(txt_path=train_txt_path, transform=trainTransform)  # 定义如何通过索引读取图片和标签
valid_data = MyDataset(txt_path=valid_txt_path, transform=validTransform)
test_data = MyDataset(txt_path=test_txt_path, transform=testTransform)

# 构建DataLoder  迭代器，把整体图片拿出来一部分放进迭代器里
train_loader = DataLoader(dataset=train_data, batch_size=train_bs, shuffle=True)  # 触发Mydataset去读取图片和标签
valid_loader = DataLoader(dataset=valid_data, batch_size=valid_bs)
test_loader = DataLoader(dataset=test_data, batch_size=test_bs)

net = Net()
net.to(device)

# ------------------------------------ step 3/5 : 定义损失函数和优化器 ------------------------------------
'''
### bmc_loss 不适用于回归问题，出现了过拟合现象，在验证集和测试集上都表现非常差
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import joblib
def bmc_loss(pred, target, noise_var):
    pred = pred.unsqueeze(dim = 1)
    target = target.unsqueeze(dim = 1)
    logits = - (pred - target.T).pow(2) / (2 * noise_var)   # logit size: [batch, batch]
    #loss = F.cross_entropy(logits, torch.arange(pred.shape[0]))     # contrastive-like loss
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).to(device))
    loss = loss * (2 * noise_var).detach()  # optional: restore the loss scale, 'detach' when noise is learnable
    loss=loss.to(torch.float)

    return loss

class BMCLoss(_Loss):
    def __init__(self, init_noise_sigma):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        return bmc_loss(pred, target, noise_var)

criterion = BMCLoss(init_noise_sigma=1.0)
'''
'''
def huber(true, pred, delta):
    loss = np.where(np.abs(true-pred) < delta , 0.5*((true-pred)**2), delta*np.abs(true - pred) - 0.5*(delta**2))
    return np.sum(loss)
'''

#criterion = torch.nn.L1Loss(reduction='mean')
#criterion = torch.nn.SmoothL1Loss()
criterion = nn.MSELoss()  # 选择损失函数
optimizer = optim.Adam(net.parameters(), lr=0.0001)  # 选择优化器
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=64)

#optimizer.add_param_group({'params': criterion.noise_sigma, 'lr': sigma_lr, 'name': 'noise_sigma'})

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)  # 设置学习率下降策略

'''
criterion = nn.MSELoss()  # 选择损失函数
optimizer = optim.AdamW(net.parameters(), lr=0.0001)  # 选择优化器
'''

# ------------------------------------ step 4/5 : 训练 --------------------------------------------------
for epoch in range(max_epoch):
    loss_sigma = 0.0  # 记录一个epoch的loss之和
    net.train()  # 让网络进入训练状态
    for data in tqdm(train_loader):
        # 获取图片和标签
        inputs, labels = data  # data相当于一个列表，取出的是一个batch的输入和标签
        labels = labels.double()
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)  # 得到模型的输出
        outputs = outputs.squeeze(-1)    # 去掉一个维度
        #outputs=outputs.to(torch.float)
        #labels=labels.to(torch.float)
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
    for data in tqdm(valid_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        outputs = outputs.squeeze(-1)
        loss = criterion(outputs, labels)
        loss_val += loss.item()
    loss_val_avg = loss_val / len(valid_loader)
    print("Epoch{:0>3}/{:0>3} Train loss:{:.4f} Valid loss:{:.4f}".format(epoch + 1, max_epoch, loss_avg, loss_val_avg))
    wind.line([[loss_avg, loss_val_avg]], [epoch + 1], win='train', update='append')

    # ------------------------------------ step5: 保存模型 ------------------------------------
    if epoch % 10 == 9:
        net_save_path = os.path.join(log_dir, 'net_params_' + str(epoch+1) + '.pkl')
        torch.save(net.state_dict(), net_save_path)
        print('Finished Training')



