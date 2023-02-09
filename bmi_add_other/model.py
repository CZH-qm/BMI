import torch
import torch.nn as nn
import timm


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # self.swin = timm.create_model('vgg16', pretrained=True)
        # self.classifer = nn.Sequential(nn.Linear(4096+128*2, 512), nn.Linear(512, 128), nn.Linear(128, 64), nn.Linear(64, 1))   #vgg16

        # self.swin = timm.create_model('resnet18', pretrained=True)
        # self.classifer = nn.Sequential(nn.Linear(512*7*7+128*2, 512), nn.Linear(512, 128), nn.Linear(128, 64), nn.Linear(64, 1))  #侧面
        # self.classifer = nn.Sequential(nn.Linear(512*7*7+128*2, 128),  nn.Linear(128, 64), nn.Linear(64, 1))  #正面

        # self.swin = timm.create_model('vit_small_patch16_224', pretrained=True)
        # self.classifer = nn.Sequential(nn.Linear(384 + 64 * 2, 64),  nn.Linear(64, 1))
        # self.classifer = nn.Sequential(nn.Linear(384 + 128 * 2, 512), nn.Linear(512, 128), nn.Linear(128, 64), nn.Linear(64, 1))

        # self.swin = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        # self.classifer = nn.Sequential(nn.Linear(1024 + 64 * 2, 64), nn.Linear(64, 1))     #swin

        self.swin = timm.create_model('convnext_small', pretrained=True)
        self.classifer = nn.Sequential(nn.Linear(768*7*7+128*2, 512),  nn.Linear(512, 128), nn.Linear(128, 64), nn.Linear(64, 1))

        # swin    vit
        # self.linear_age = nn.Sequential(nn.Linear(4, 64))
        # self.linear_gender = nn.Sequential(nn.Linear(2, 64))

        # self.linear_age=nn.Linear(4,128)
        # self.linear_gender=nn.Linear(2,128)

        #resnet18   vgg16   convnext
        self.linear_age = nn.Sequential(nn.Linear(4, 64), nn.Linear(64, 128))
        self.linear_gender = nn.Sequential(nn.Linear(2, 64), nn.Linear(64, 128))


    def forward(self, x, age, gender):
        features = self.swin.forward_features(x)
        age = age.reshape(len(age), -1)
        gender = gender.reshape(len(gender), 2)
        age = self.linear_age(torch.tensor(age, dtype=torch.float64))
        gender = self.linear_gender(torch.tensor(gender, dtype=torch.float64))
        features = features.contiguous().view(-1, 768*7*7)
        # features=features.view(-1,512*7*7)   #resnet18
        features = torch.cat((features, age, gender), 1)
        output = self.classifer(features)
        return output


if __name__ == "__main__":
    model = Net()
    input = torch.randn(2, 3, 224, 224)
    out = model(input)
    print(out)
