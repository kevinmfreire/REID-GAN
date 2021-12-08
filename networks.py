import torch.nn as nn
import torch
import torch.nn.functional as F
from math import exp
from torch.autograd import Variable
import numpy as np

class GNet(nn.Module):
    def __init__(self, image_size):
        super(GNet, self).__init__()
        self.input_channel=1
        cnum = 32
        size = image_size//2
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, cnum, 9, 1, padding=4),
            nn.BatchNorm2d(cnum),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(cnum, 2*cnum, 3, 1, padding=1),
            nn.BatchNorm2d(2*cnum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2*cnum, 4*cnum, 3, 1, padding=1),
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
            nn.Conv2d(cnum, 1, 9, 1, padding=4),
            nn.BatchNorm2d(self.input_channel),
            nn.Tanh()
        )

        self.norm = nn.BatchNorm2d(self.input_channel)

        self.pool = nn.MaxPool2d(kernel_size=(2,2))

    
    def forward(self, input):

        # img = self.norm(input)

        conv1 = self.conv1(input)
        
        pool1 = self.pool(conv1)
        
        conv2 = self.conv2(pool1)

        pool2 = self.pool(conv2)

        res1 = self.residual(pool2)

        x = res1.add(pool2)

        res2 = self.residual(x)

        x = res2.add(x)

        res3 = self.residual(x)

        x = res3.add(x)

        deconv1 = self.deconv(x)

        x = deconv1.add(conv1)

        conv3 = self.conv3(x)

        output = conv3.add(input)

        x = output + 1.0

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
