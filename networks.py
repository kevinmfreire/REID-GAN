import torch.nn as nn
import torch
import torch.nn.functional as F
from math import exp
from torch.autograd import Variable
import numpy as np

class GNet(nn.Module):
    def __init__(self):
        super(GNet, self).__init__()
        self.input_channel = 1
        self.inter_channel = 64

        self.conv1 = nn.Sequential(nn.Conv2d(self.input_channel, self.inter_channel, 3, 1, padding=1),
                                    nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(nn.Conv2d(self.inter_channel, self.inter_channel*2, 3, 2, padding=1),
                                    nn.BatchNorm2d(self.inter_channel*2),
                                    nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(nn.Conv2d(self.inter_channel*2, self.inter_channel*4, 3, 2, padding=1),
                                    nn.BatchNorm2d(self.inter_channel*4),
                                    nn.ReLU(inplace=True))
        self.residual = nn.Sequential(nn.Conv2d(4*self.inter_channel, 4*self.inter_channel, 3, 1,padding=1),
                                    nn.BatchNorm2d(4*self.inter_channel),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(4*self.inter_channel, 4*self.inter_channel, 3, 1,padding=1),
                                    nn.BatchNorm2d(4*self.inter_channel))
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(self.inter_channel*4, self.inter_channel*4, 4, 2, padding=1),
                                    nn.BatchNorm2d(self.inter_channel*4),
                                    nn.ReLU(inplace=True))
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(self.inter_channel*6, self.inter_channel*2, 4, 2, padding=1),
                                    nn.BatchNorm2d(self.inter_channel*2),
                                    nn.ReLU(inplace=True))
        self.conv2 = nn.Conv2d(self.inter_channel*3, self.input_channel, 3, padding=1)
    
    def forward(self, input):

        conv1 = self.conv1(input)

        layer1 = self.layer1(conv1)

        layer2 = self.layer2(layer1)

        res1 = self.residual(layer2)

        sum1 = res1.add(layer2)

        res2 = self.residual(sum1)

        sum2 = res2.add(sum1)

        res3 = self.residual(sum2)

        sum3 = res3.add(sum2)

        res4 = self.residual(sum3)

        sum4 = res4.add(sum3)

        res5 = self.residual(sum4)

        sum5 = res5.add(sum4)

        deconv1 = self.deconv1(sum5)

        deconv2 = self.deconv2(torch.cat((layer1,deconv1),1))

        output = self.conv2(torch.cat((conv1,deconv2),1))

        return output

class DNet(nn.Module):
    def __init__(self):
        super(DNet, self).__init__()
        self.input_channel = 1
        self.inter_channel = 64

        self.conv1 = nn.Sequential(nn.Conv2d(self.input_channel, self.inter_channel, 3, 2, padding =1),
                                   nn.LeakyReLU(0.2, inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(self.inter_channel, 2*self.inter_channel, 3, 2, padding=1),
                                    nn.BatchNorm2d(2*self.inter_channel),
                                    nn.LeakyReLU(0.2, inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(2*self.inter_channel, 4*self.inter_channel, 3, 2, padding=1),
                                    nn.BatchNorm2d(4*self.inter_channel),
                                    nn.LeakyReLU(0.2, inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(4*self.inter_channel, 8*self.inter_channel, 3, 2, padding=1),
                                    nn.BatchNorm2d(8*self.inter_channel),
                                    nn.LeakyReLU(0.2, inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(8*self.inter_channel, 16*self.inter_channel, 3, 2, padding=1),
                                    nn.BatchNorm2d(16*self.inter_channel),
                                    nn.Sigmoid())

    def forward(self, input):

        conv1 = self.conv1(input)

        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)

        conv4 = self.conv4(conv3)

        output = self.conv5(conv4)

        return output
