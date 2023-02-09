from model import Net
import torch
from utils import MyDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import os
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


# 正面
# test_txt_path = r'C:\Users\Administrator\Desktop\diannao\bmi\newData\zhedang\bottom1_4\bottom_1_4.csv'
# model_name = 'bottom_1_4_vit'

test_txt_path = r'C:\Users\Administrator\Desktop\diannao\bmi\newData\zhedang\bottom1_2\bottom_1_2.csv'
model_name = 'bottom_1_2_convnext'

# test_txt_path = r'C:\Users\Administrator\Desktop\diannao\bmi\newData\zhedang\top1_2\top_1_2.csv'
# model_name = 'top_1_2_vit'

# 侧面
# test_txt_path = r'C:\Users\Administrator\Desktop\diannao\bmi\newData\zhedang\top1_2\ce_top_1_2.csv'
# model_name = 'ce_top_1_2_vit'


# test_txt_path = r'C:\Users\Administrator\Desktop\diannao\bmi\newData\zhedang\bottom1_2\ce_bottom_1_2.csv'
# model_name = 'ce_bottom_1_2_vit'

# test_txt_path = r'C:\Users\Administrator\Desktop\diannao\bmi\newData\zhedang\bottom1_4\ce_bottom_1_4.csv'
# model_name = 'ce_bottom_1_4_vit'

test_bs = 4

normMean = [0.7522175, 0.7522175, 0.7522175]
normStd = [0.20582062, 0.20582062, 0.20582062]
normTransform = transforms.Normalize(normMean, normStd)  # 归一化的公式
testTransform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    normTransform
])
test_data = MyDataset(txt_path=test_txt_path, transform=testTransform)
test_loader = DataLoader(dataset=test_data, batch_size=test_bs)

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
net = Net()
net.to(device)  # 创建一个网络 （实例化一个模型）

pthfile = r'E:\my_code\结果\预测bmi\平面\convnext\06-10_21-52-07\net_params_100.pkl'  # 加载模型
#pthfile = r'E:\my_code\bmi_code\Result\05-01_16-31-28\net_params_100.pkl'

net.load_state_dict(torch.load(pthfile))
#
# def bmc_loss(pred, target, noise_var):
#     pred = pred.unsqueeze(dim = 1)
#     target = target.unsqueeze(dim = 1)
#     logits = - (pred - target.T).pow(2) / (2 * noise_var)   # logit size: [batch, batch]
#     #loss = F.cross_entropy(logits, torch.arange(pred.shape[0]))     # contrastive-like loss
#     loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).to(device))
#     loss = loss * (2 * noise_var).detach()  # optional: restore the loss scale, 'detach' when noise is learnable
#     loss=loss.to(torch.float)
#     return loss
#
# class BMCLoss(_Loss):
#     def __init__(self, init_noise_sigma):
#         super(BMCLoss, self).__init__()
#         self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))
#
#     def forward(self, pred, target):
#         noise_var = self.noise_sigma ** 2
#         return bmc_loss(pred, target, noise_var)

loss_sigma = 0.0
net.eval()  # 进入验证状态
#criterion = BMCLoss(init_noise_sigma=1.0)
criterion = nn.MSELoss()
mse_all = []
mae_all = []
output_all = []
label_all = []
for i, data in enumerate(test_loader):
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)

    # forward, backward, update weights

    outputs = net(inputs)  # 得到模型的输出

    # 计算loss
    mse = mean_squared_error(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
    mae = mean_absolute_error(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
    #output_all.append(outputs.cpu().detach().numpy()[0][0])
    output_all += [j[0] for j in outputs.cpu().detach().numpy()]
    label_all.append(labels.cpu().detach().numpy()[0])
    mse_all.append(mse)
    mae_all.append(mae)
print('the average MSE is ', np.mean(mse_all))
print('the average MAE is ', np.mean(mae_all))
save_path = 'test_result'
if not os.path.exists(save_path):
    os.makedirs(save_path)
# data2 = {'label': label_all}
data2 = {'output': output_all}
# data2 = {'label': label_all,'output': output_all}  # , 'predict': test_save_output,'score':score_new_test, }
df2 = pd.DataFrame(data2)
df2.to_csv(os.path.join(save_path, model_name + '_test.csv'), index=None)
print('Finished saving csvfile!')
