import torch.nn as nn
import torch
import torch.nn.functional as F
from math import exp
from torch.autograd import Variable
import numpy as np

class GNet(nn.Module):
    def __init__(self):
        super(GNet, self).__init__()
        self.input_channel=1
        cnum = 32
        size = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, cnum, 9, 1,padding=4),
            nn.BatchNorm2d(cnum),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(cnum, 2*cnum, 3, 1),
            nn.BatchNorm2d(2*cnum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2*cnum, 4*cnum, 3, 1),
            nn.BatchNorm2d(4*cnum),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.residual = nn.Sequential(
            nn.Conv2d(4*cnum, 4*cnum, 3, 1,padding=1),
            nn.BatchNorm2d(4*cnum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4*cnum, 4*cnum, 3, 1,padding=1),
            nn.BatchNorm2d(4*cnum),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.deconv = nn.Sequential(
            nn.Upsample(size, mode='nearest'),
            nn.Conv2d(4*cnum, 2*cnum, 3, 1),
            nn.BatchNorm2d(2*cnum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(2*size, mode='nearest'),
            nn.Conv2d(2*cnum, cnum, 3, 1,padding=1),
            nn.BatchNorm2d(cnum),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(cnum, 1, 9, 1,padding=4),
            nn.BatchNorm2d(self.input_channel),
            nn.Tanh()
        )

    
    def forward(self, input):

        x1 = self.conv1(input)

        x = self.conv2(x1)

        x2 = self.residual(x)

        x = x2.add(x)

        x3 = self.residual(x)

        x = x3.add(x)

        x4 = self.residual(x)

        x = x4.add(x)

        x5 = self.deconv(x)

        x = x5.add(x1)

        x = self.conv3(x)

        x = x + 1.0

        x = x/2.0

        return x

class DNet(nn.Module):
    def __init__(self):
        super(DNet, self).__init__()
        self.input_channel=1
        cnum = 48
        self.model = nn.Sequential(
            nn.Conv2d(1, cnum, 4, 2),
            nn.BatchNorm2d(cnum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cnum, 2*cnum, 4, 2),
            nn.BatchNorm2d(2*cnum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2*cnum, 4*cnum, 4, 2),
            nn.BatchNorm2d(4*cnum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4*cnum, 8*cnum, 4, 1),
            nn.BatchNorm2d(8*cnum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8*cnum, 1, 4, 1),
            nn.BatchNorm2d(self.input_channel),
            nn.Sigmoid()
        )

    def forward(self, img):
        img = self.model(img)
        return img