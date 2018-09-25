import torch.nn as nn
import torch
import torch.nn.functional as F

class D_Net(nn.Module):
    def __init__(self):
        super(D_Net, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True))
        self.conv5 = nn.Sequential(nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True))
        self.fc1 = nn.Linear(3*3*256, 512, bias=True)
        self.fc2 = nn.Linear(512, 1, bias=True)
        self.downsample = nn.MaxPool2d(2, 2)

    def forward(self, f, training=True):
        f = self.conv1(f)
        f = self.conv2(self.downsample(f))
        f = self.conv3(self.downsample(f))
        f = self.conv4(self.downsample(f))
        f = self.conv5(self.downsample(f))
        f = self.downsample(f)
        #print(f.size())
        f = f.view(f.size(0), -1)
        f = F.relu(self.fc1(f))
        p = torch.sigmoid(self.fc2(f))
        return p
