import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import timm


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # self.pretrained_model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        self.pretrained_model = timm.create_model('convnext_small', pretrained=True)
        # self.pretrained_model = timm.create_model('vgg16', pretrained=True)
        # self.pretrained_model = timm.create_model('vit_small_patch16_224', pretrained=True)
        # self.pretrained_model = timm.create_model('resnet18', pretrained=True)

        self.classifer = nn.Sequential(
            # nn.Linear(512*7*7, 1024), nn.Linear(1024, 128), nn.Linear(128, 64), nn.Linear(64, 1)) # resnet18
            nn.Linear(768*7*7, 1024), nn.Linear(1024, 128), nn.Linear(128, 64), nn.Linear(64, 1))    #convnext
            # nn.Linear(1024, 256), nn.Linear(256, 128), nn.Linear(128, 64), nn.Linear(64, 1))      #swin   w ziji
            # nn.Linear(1024, 128), nn.Linear(128, 64), nn.Linear(64, 1))      #swin  sunrong
            # nn.Linear(384, 64), nn.Linear(64, 1))    #vit
            # nn.Linear(4096, 1024), nn.Linear(1024, 128), nn.Linear(128, 64), nn.Linear(64, 1))   #vgg16


    def forward(self, x):
        features = self.pretrained_model.forward_features(x)
        features = features.contiguous().view(features.size(0), -1)
        output = self.classifer(features)
        return output



if __name__ == "__main__":
    model = Net()
    input = torch.randn(10, 3, 224, 224)
    out = model(input)

    print(out.shape)
