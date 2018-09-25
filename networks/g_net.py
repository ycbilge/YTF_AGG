import torch.nn as nn
import torch

read_img_count = 20

class G_Net(nn.Module):
    def __init__(self):
        super(G_Net, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(read_img_count*3, 64, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), nn.PReLU())#aggN*3
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(128), nn.PReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(256), nn.PReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(256, 128, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(128), nn.PReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(128, 64, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), nn.PReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), nn.PReLU())
        self.conv6_p = nn.Sequential(nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), nn.PReLU())
        self.conv7 = nn.Sequential(nn.Conv2d(64, 3, 3, stride=1, padding=1, bias=False), nn.Sigmoid())
        self.downsample = nn.MaxPool2d(2, 2)
        self.upsample = torch.nn.UpsamplingBilinear2d(scale_factor=2)


    def forward(self, f, training=True):
        f1 = self.conv1(f)
        f2 = self.conv2(self.downsample(f1))
        f3 = self.conv3(self.downsample(f2))
        f4 = self.conv4(self.upsample(f3))
        f5 = self.conv5(self.upsample(f4))
        f_mid = f1 + f5
        f6 = self.conv6(f_mid)
        f6 = self.conv6_p(f6)
        f7 = self.conv7(f6)
        return f7