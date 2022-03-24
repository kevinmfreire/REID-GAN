import torch.nn as nn
import torch
import torch.nn.functional as F
from math import exp
from torch.autograd import Variable
import numpy as np
from typing import Callable, Any, Optional, Tuple, List
import warnings
    
class ImageDiscriminator(nn.Module):
    """
    Discriminator Network
    """
    def __init__(self):
        super(ImageDiscriminator, self).__init__()
        self.inter_channels = 64
        conv_block = SNConv2d

        self.conv1 = conv_block(1, self.inter_channels, 3, 1, padding=1)
        self.conv2 = conv_block(self.inter_channels, self.inter_channels*2, 3, 2, padding=1)
        self.conv3 = conv_block(self.inter_channels*2, self.inter_channels*4, 3, 2, padding=1)
        self.conv4 = conv_block(self.inter_channels*4, self.inter_channels*8, 3, 2, padding=1)
        self.conv5 = conv_block(self.inter_channels*8, self.inter_channels*8, 3, 2, padding=1)
        
    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv5(x)
        x = x.view((x.size(0), -1))

        return x

class RIGAN(nn.Module):
    """
    Residual Inception Generative Adversarial Network (RIGAN)
    """
    def __init__(self, inception_blocks: Optional[List[Callable[..., nn.Module]]] = None):
        super(RIGAN, self).__init__()
        if inception_blocks is None:
            inception_blocks = [BasicConv2d, InceptionA, InceptionB, InceptionC, InceptionD, InceptionE, BasicDeconv2d]
        
        self.in_channels = 64
        self.inter_channels = 32
        
        assert len(inception_blocks) == 7
        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]
        deconv_block = inception_blocks[6]

        self.channel_conv1 = conv_block(1, self.inter_channels, 1, 1)
        self.spatial_conv1 = conv_block(self.inter_channels, self.inter_channels, 3, 2, padding=1)
        self.channel_conv2 = conv_block(self.inter_channels, self.inter_channels*2, 1, 1)
        self.spatial_conv2 = conv_block(self.inter_channels*2, self.inter_channels*2, 3, 2, padding=1)

        self.layer_block_1 = self._make_layer(inception_a, 2, inter_channels=32)
        self.reduce_grid_1 = self._alter_grid_layer(inception_b, 128, inter_channels=64)
        self.layer_block_2 = self._make_layer(inception_c, 2, inter_channels=64, channels_7x7=64)
        self.reduce_grid_2 = self._alter_grid_layer(inception_d, 256, inter_channels=128)
        self.layer_block_3 = self._make_layer(inception_e, 1, inter_channels=256)
        self.expand_grid_1 = self._alter_grid_layer(inception_d, 1024, inter_channels=128, decoder=1)
        self.layer_block_4 = self._make_layer(inception_c, 2, inter_channels=64, channels_7x7=64)
        self.expand_grid_2 = self._alter_grid_layer(inception_b, 256, inter_channels=64, decoder=1)
        self.layer_block_5 = self._make_layer(inception_a, 2, inter_channels=32)

        self.channel_deconv1 = deconv_block(192, self.inter_channels*2, 1, 1)
        self.spatial_deconv1 = deconv_block(self.inter_channels*2, self.inter_channels*2, 4, 2, padding=1)
        self.channel_deconv2 = deconv_block(96, self.inter_channels, 1, 1)
        self.spatial_deconv2 = deconv_block(self.inter_channels, self.inter_channels, 4, 2, padding=1)

        self.out_conv = nn.Conv2d(33, 1, 1, bias=True)

    def forward(self, x):
        
        identity = x.clone()
        x = self.channel_conv1(x)
        conv1 = self.spatial_conv1(x)
        x = self.channel_conv2(conv1)
        conv2 = self.spatial_conv2(x)

        x = self.layer_block_1(conv2)
        x = self.reduce_grid_1(x)
        x = self.layer_block_2(x)
        x = self.reduce_grid_2(x)
        x = self.layer_block_3(x)
        x = self.expand_grid_1(x)
        x = self.layer_block_4(x)
        x = self.expand_grid_2(x)
        x = self.layer_block_5(x)

        x = self.channel_deconv1(torch.cat((conv2,x),1))
        x = self.spatial_deconv1(x)
        x = self.channel_deconv2(torch.cat((conv1,x),1))
        x = self.spatial_deconv2(x)
        out = self.out_conv(torch.cat((identity,x),1))

        return out

    def _make_layer(self, ri_block: Optional[Callable[..., nn.Module]], num_residual_blocks, inter_channels, channels_7x7=None):
        identity_downsample = None
        layers = []

        if self.in_channels != inter_channels*4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, inter_channels*4, 1, 1, bias=False),
                nn.BatchNorm2d(inter_channels*4),
                nn.ReLU()
            )

        if channels_7x7 is not None:
            
            layers.append(
            ri_block(self.in_channels, channels_7x7, identity_downsample)
            )

            self.in_channels = inter_channels*4

            for i in range(num_residual_blocks - 1):
                layers.append(ri_block(self.in_channels, channels_7x7))
        else:
            layers.append(
                ri_block(self.in_channels, identity_downsample)
            )

            self.in_channels = inter_channels*4

            for i in range(num_residual_blocks - 1):
                layers.append(ri_block(self.in_channels))

        return nn.Sequential(*layers)

    def _alter_grid_layer(self, inception_block: Optional[Callable[..., nn.Module]], in_channels, inter_channels, decoder=None):
        self.in_channels = inter_channels*4
        return inception_block(in_channels, decoder)

class SNConv2d(nn.Module):
    """
    SN convolution for spetral normalization conv
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation = 1, groups=1, bias=True):
        super(SNConv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv2d = nn.utils.spectral_norm(self.conv2d)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
    def forward(self,x):
        x = self.conv2d(x)
        if self.activation is not None:
            return self.activation(x)
        else:
            return x

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.activation(x)

class BasicDeconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0):
        super(BasicDeconv2d, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        return self.activation(x)

class InceptionA(nn.Module):
    def __init__(self, in_channels, identity_downsample=None, conv_block: Optional[Callable[..., nn.Module]] = None):
        super(InceptionA, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 32, kernel_size=1)

        self.branch3x3_1 = conv_block(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = conv_block(16, 32, kernel_size=3, padding=1)

        self.branch3x3dbl_1 = conv_block(in_channels, 32, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(32, 48, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(48, 48, kernel_size=3, padding=1)

        self.branch_pool = conv_block(in_channels, 16, kernel_size=1)

        self.identity_downsample = identity_downsample

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        identity = x.clone()

        outputs = self._forward(x)
        outputs = torch.cat(outputs,1)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        outputs += identity

        return outputs

class InceptionB(nn.Module):
    def __init__(self, in_channels, conv_block: Optional[Callable[..., nn.Module]] = None, decoder=None):
        super().__init__()
        if conv_block is None and decoder is None:
            conv_block = BasicConv2d
            kernel = 3
            self.sample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            conv_block = BasicDeconv2d
            kernel = 4
            self.sample = nn.Sequential(
                nn.Conv2d(in_channels, 128, kernel_size=1, bias=False),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            )

        self.branch3x3_1 = conv_block(in_channels, 32, kernel_size=1)
        self.branch3x3_2 = conv_block(32, 64, kernel_size=kernel, stride=2, padding=1)

        self.branch3x3dbl_1 = conv_block(in_channels, 32, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(32, 48, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(48, 64, kernel_size=kernel, stride=2, padding=1)

    def _forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.sample(x)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class InceptionC(nn.Module):
    def __init__(
        self, in_channels, channels_7x7, identity_downsample=None, conv_block: Optional[Callable[..., nn.Module]] = None):
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 64, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = conv_block(c7, 64, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = conv_block(c7, 64, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = conv_block(in_channels, 64, kernel_size=1)

        self.identity_downsample = identity_downsample

    def _forward(self, x):
        
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return outputs

    def forward(self, x):
        identity = x.clone()
        
        outputs = self._forward(x)
        outputs = torch.cat(outputs,1)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        outputs += identity

        return outputs

class InceptionD(nn.Module):
    def __init__(self, in_channels, conv_block: Optional[Callable[..., nn.Module]] = None, decoder=None):
        super().__init__()
        if conv_block is None and decoder is None:
            conv_block = BasicConv2d
            kernel = 3
            self.sample = nn.MaxPool2d(kernel_size=3, stride=2, padding = 1)
        else:
            conv_block = BasicDeconv2d
            kernel = 4
            self.sample = nn.Sequential(
                nn.Conv2d(in_channels, 256, kernel_size=1, bias=False),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            )

        self.branch3x3_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3_2 = conv_block(64, 128, kernel_size=kernel, stride=2, padding = 1)

        self.branch7x7x3_1 = conv_block(in_channels, 128, kernel_size=1)
        self.branch7x7x3_2 = conv_block(128, 128, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = conv_block(128, 128, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = conv_block(128, 128, kernel_size=kernel, stride=2, padding = 1)

    def _forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        
        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)
        
        branch_pool = self.sample(x)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class InceptionE(nn.Module):
    def __init__(self, in_channels, identity_downsample=None, conv_block: Optional[Callable[..., nn.Module]] = None):
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 160, kernel_size=1)

        self.branch3x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch3x3_2a = conv_block(192, 192, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = conv_block(192, 192, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = conv_block(in_channels, 224, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(224, 192, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(192, 192, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = conv_block(192, 192, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = conv_block(in_channels, 96, kernel_size=1)

        self.identity_downsample = identity_downsample

    def _forward(self, x):

        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        identity = x.clone()

        outputs = self._forward(x)
        outputs = torch.cat(outputs,1)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        outputs += identity

        return outputs