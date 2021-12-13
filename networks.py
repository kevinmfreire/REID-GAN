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
        self.conv2 = nn.Sequential(nn.Conv2d(self.inter_channel*3, self.input_channel, 3, padding=1),
                                    nn.Tanh())
    
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

        res4 = self.residual(res3)

        sum4 = res4.add(sum3)

        deconv1 = self.deconv1(sum4)

        deconv2 = self.deconv2(torch.cat((layer1,deconv1),1))

        output = self.conv2(torch.cat((conv1,deconv2),1))

        return output

class DNet(nn.Module):
    def __init__(self, batch_size, patch_size, patch_n):
        super(DNet, self).__init__()
        self.input_channel = 1
        self.inter_channel = 64
        self.size = (patch_size//8)**2
        self.batch_size = batch_size
        self.patch_n = patch_n

        self.layer1 = nn.Sequential(nn.Conv2d(self.input_channel, self.inter_channel, 3, 1),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Conv2d(self.inter_channel, self.inter_channel, 3, 2, padding=2),
                                    nn.BatchNorm2d(self.inter_channel),
                                    nn.LeakyReLU(0.2, inplace=True))
        self.layer2 = nn.Sequential(nn.Conv2d(self.inter_channel, 2*self.inter_channel, 3, 1),
                                    nn.BatchNorm2d(2*self.inter_channel),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Conv2d(2*self.inter_channel, 2*self.inter_channel, 3, 2, padding=2),
                                    nn.BatchNorm2d(2*self.inter_channel),
                                    nn.LeakyReLU(0.2, inplace=True))
        self.layer3 = nn.Sequential(nn.Conv2d(2*self.inter_channel, 4*self.inter_channel, 3, 1),
                                    nn.BatchNorm2d(4*self.inter_channel),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Conv2d(4*self.inter_channel, 4*self.inter_channel, 3, 2, padding=2),
                                    nn.BatchNorm2d(4*self.inter_channel),
                                    nn.LeakyReLU(0.2, inplace=True))
        self.drop_out = nn.Dropout()
        self.fc_layer = nn.Sequential(nn.Linear(4*self.inter_channel*self.size, 1024),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(1024,1))
        # self.linear2 = nn.Linear(batch_size*patch_n,1)

    def forward(self, input):

        layer1 = self.layer1(input)

        layer2 = self.layer2(layer1)

        layer3 = self.layer3(layer2)

        x = layer3.view(layer3.size(0),-1)

        drop_out = self.drop_out(x)

        fc1 = self.fc_layer(drop_out)
        
        # x2 = fc1.flatten()

        # fc2 = self.linear2(x2)

        return fc1
