import numpy as np
import torch
import tqdm
from utils import MyDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import timm
from sklearn.metrics import confusion_matrix
import seaborn
import matplotlib.pyplot as plt


def plot(matrix):
    seaborn.set()
    f, ax = plt.subplots()
    print(matrix)
    seaborn.heatmap(matrix, annot=True, cmap="Blues", ax=ax)  # 画热力图
    ax.set_title('confusion matrix')  # 标题
    ax.set_xlabel('predict')  # x轴
    ax.set_ylabel('true')  # y轴
    plt.show()

# 正面
# test_path = r'C:\Users\Administrator\Desktop\diannao\bmi\newData\zhedang\bottom1_4\bottom_1_4.csv'
# model_name = 'bottom_1_4_convnext'

# test_path = r'C:\Users\Administrator\Desktop\diannao\bmi\newData\zhedang\bottom1_2\bottom_1_2.csv'
# model_name = 'bottom_1_2_convnext'

# test_path = r'C:\Users\Administrator\Desktop\diannao\bmi\newData\zhedang\top1_2\top_1_2.csv'
# model_name = 'top_1_2_convnext'

# 侧面
test_path = r'C:\Users\Administrator\Desktop\diannao\bmi\newData\zhedang\top1_2\ce_top_1_2.csv'
model_name = 'ce_top_1_2_vit'

# test_path = r'C:\Users\Administrator\Desktop\diannao\bmi\newData\zhedang\bottom1_2\ce_bottom_1_2.csv'
# model_name = 'ce_bottom_1_2_convnext'

# test_path = r'C:\Users\Administrator\Desktop\diannao\bmi\newData\zhedang\bottom1_4\ce_bottom_1_4.csv'
# model_name = 'ce_bottom_1_4_convnext'

test_bs = 1

normMean = [0.7522175, 0.7522175, 0.7522175]
normStd = [0.20582062, 0.20582062, 0.20582062]
normTransform = transforms.Normalize(normMean, normStd)  # 归一化的公式
testTransform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    normTransform
])
test_data = MyDataset(txt_path=test_path, transform=testTransform)
test_loader = DataLoader(dataset=test_data, batch_size=test_bs)

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

# net = timm.create_model('resnet18', pretrained=False, num_classes=2)
# net = timm.create_model('vit_small_patch16_224', pretrained=True)
# net = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
# net = timm.create_model('vgg16', pretrained=True)
net = timm.create_model('convnext_small', pretrained=False, num_classes=2)


net.to(device)

pthfile = r'E:\my_code\结果\预测性别\侧面\convnext\06-28_22-50-56\net_params_100.pkl'  # 加载模型

net.load_state_dict(torch.load(pthfile))

net.eval()  # 进入验证状态
Y_true = []
Y_pred = []
for data in tqdm.tqdm(test_loader):
    inputs, labels = data
    inputs = inputs.to(device)
    outputs = net(inputs)
    outputs = np.argmax(outputs.cpu().detach())
    Y_true.append(int(labels))

    Y_pred.append(int(outputs))

matrix = confusion_matrix(y_true=Y_true, y_pred=Y_pred)
plot(matrix)
