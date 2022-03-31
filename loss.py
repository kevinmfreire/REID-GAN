import torch.nn as nn
import torch
import torch.nn.functional as F
import argparse
from math import exp
from torch.autograd import Variable
from torch import linalg
import torchvision.models as models
from torchvision.models._utils import IntermediateLayerGetter
from torch.nn.modules.loss import _Loss

# Check CUDA's presence
cuda_is_present = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda_is_present else torch.FloatTensor

def to_cuda(data):
    	return data.cuda() if cuda_is_present else data

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window =_1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return  to_cuda(window)

def mse_loss(target, prediction):
    _, C, H, W = target.size()
    N = C*H*W
    pixel_difference = target - prediction
    pixel_loss = pixel_difference.norm(p=2) / float(N)
    return pixel_loss

# OBTAIN VGG16 PRETRAINED MODEL EXCLUDING FULLY CONNECTED LAYERS
def get_feature_layer_vgg16(image, layer, model):
    image = torch.cat([image,image,image],1)
    return_layers = {'{}'.format(layer): 'feat_layer_{}'.format(layer)}
    output_feature = IntermediateLayerGetter(model.features, return_layers=return_layers)
    image_feature = output_feature(image)
    return image_feature['feat_layer_{}'.format(layer)]

def compute_SSIM(img1, img2, window_size, channel, size_average=True):
    # referred from https://github.com/Po-Hsun-Su/pytorch-ssim
    if len(img1.size()) == 2:
        shape_ = img1.shape[-1]
        img1 = img1.view(1,1,shape_ ,shape_ )
        img2 = img2.view(1,1,shape_ ,shape_ )
    window = create_window(window_size, channel)
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size//2)
    mu2 = F.conv2d(img2, window, padding=window_size//2)
    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2) - mu1_mu2

    C1, C2 = 0.01**2, 0.03**2

    ssim_map = ((2*mu1_mu2+C1)*(2*sigma12+C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class Vgg16FeatureExtractor(nn.Module):
    
    def __init__(self, layers=[3, 8, 15, 22, 29],pretrained=False, progress=True, **kwargs):
        super(Vgg16FeatureExtractor, self).__init__()
        self.layers=layers
        self.model = models.vgg16(pretrained, progress, **kwargs)
        del self.model.avgpool
        del self.model.classifier
        self.return_layers = {'{}'.format(self.layers[i]): 'feat_layer_{}'.format(self.layers[i]) for i in range(len(self.layers))}
        self.model = IntermediateLayerGetter(self.model.features, return_layers=self.return_layers)

    def forward(self, x):
        feats = list()
        out = self.model(x)     
        for i in range(len(self.layers)):
            feats.append(out['feat_layer_{}'.format(self.layers[i])])
        return feats

class PerceptualLoss(_Loss):
    
    def __init__(self, blocks=[1, 2, 3, 4, 5], mse_weight=1, vgg_weight=0.02):
        super(PerceptualLoss, self).__init__()

        self.mse_weight = mse_weight
        self.vgg_weight = vgg_weight

        self.blocks = blocks
        self.model = Vgg16FeatureExtractor(pretrained=True)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

        self.perceptual = nn.MSELoss(reduction='sum')
        # self.mse = nn.MSELoss()

    def forward(self, input, target):
        loss_value = 0

        input_feats = self.model(torch.cat([input, input, input], dim=1))
        target_feats = self.model(torch.cat([target, target, target], dim=1))

        feats_num = len(self.blocks)
        for idx in range(feats_num):
            input, target = input_feats[idx], target_feats[idx]
            _, d, w, h = target.size()
            feat_count = float(d*w*h)
            loss_value += self.perceptual(input, target) / feat_count
        loss_value /= feats_num
        
        # loss = self.mse_weight * self.mse(input, target) + self.resnet_weight * loss_value
        loss = self.vgg_weight * loss_value

        return loss

class MPL(torch.nn.Module):
    """
    The Multi Perceptual Loss Function
    """
    def __init__(self):
        super(MPL, self).__init__()
        self.model = models.vgg16(pretrained=True)
        self.model.to(torch.device('cuda' if cuda_is_present else 'cpu'))
        self.model.eval()
    def forward(self, target, prediction):
        perceptual_loss = 0
        vgg19_layers = [3, 8, 15, 22, 29] # layers: 3, 8, 15, 22, 29
        for layer in vgg19_layers:
            feature_target = get_feature_layer_vgg16(target, layer, self.model)
            feature_prediction = get_feature_layer_vgg16(prediction, layer, self.model)
            _, _, H, W = feature_target.size()
            feature_count = W*H
            feature_difference = feature_target - feature_prediction
            # feature_loss = feature_difference.norm(dim=1, p=2) / float(feature_count)
            feature_loss = linalg.norm(feature_difference, dim=1, ord=2) / float(feature_count)
            perceptual_loss += feature_loss.mean()
        return perceptual_loss

class SSIM(_Loss):
    """
    The Dissimilarity Loss funciton
    """
    def __init__(self, window_size = 11, size_average = True, ssim_weight = 0.48):
        super(SSIM, self).__init__()
        self.ssim_weight = ssim_weight
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        self.window.to(torch.device('cuda' if cuda_is_present else 'cpu'))

    def forward(self, y, pred):
        ssim = compute_SSIM(y, pred, self.window_size, self.channel, self.size_average)
        dssim = (1.0 - ssim) / 2.0
        return self.ssim_weight * dssim

class DLoss(_Loss):
    """
    The loss for discriminator
    """
    def __init__(self):
        super(DLoss, self).__init__()
        self.activation = nn.ReLU()
        
    def forward(self, Dy, Dg):
        # return self.activation(1-torch.mean(Dy)) + self.activation(1+torch.mean(Dg))
        return -torch.mean(Dy) + torch.mean(Dg)

class GLoss(_Loss):
    """
    The loss for generator
    """
    def __init__(self, weight=0.50):
        super(GLoss, self).__init__()
        self.weight = weight

    def forward(self, Dg):
        return -self.weight * torch.mean(Dg)