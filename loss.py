import torch.nn as nn
import torch
import torch.nn.functional as F
import argparse
from math import exp
from torch.autograd import Variable
import torchvision.models as models
from torchvision.models._utils import IntermediateLayerGetter

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

def get_feature_loss(target, prediction, layer, model):
    feature_transformed_target = get_feature_layer_vgg16(target, layer, model)
    feature_transformed_prediction = get_feature_layer_vgg16(prediction, layer, model)
    _, C, H, W = feature_transformed_target.size()
    feature_count = C*W*H
    feature_difference = feature_transformed_prediction - feature_transformed_target
    feature_loss = feature_difference.norm(p=2) / float(feature_count)
    return feature_loss

def get_smooth_loss(image):
    _, _ , image_height, image_width = image.size()
    horizontal_normal = image[:, :, :, 0:image_width-1]
    horizontal_one_right = image[:, :, :, 1:image_width]
    vertical_normal = image[:, :, 0:image_height-1, :]
    vertical_one_right = image[:, :, 1:image_height, :]
    # smooth_loss = torch.sum(torch.pow(horizontal_normal-horizontal_one_right, 2)) / 2.0 + torch.sum(torch.pow(vertical_normal - vertical_one_right, 2)) / 2.0
    smooth_loss = mse_loss(horizontal_normal, horizontal_one_right) + mse_loss(vertical_normal, vertical_one_right) 
    return smooth_loss

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

class STD(torch.nn.Module):
    def __init__(self, window_size = 5):
        super(STD, self).__init__()
        self.window_size = window_size
        self.channel=1
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.window=create_window(self.window_size, self.channel)
        self.window.to(torch.device('cuda' if cuda_is_present else 'cpu'))
    def forward(self, img):
        mu = F.conv2d(img, self.window, padding = self.window_size//2, groups = self.channel)
        mu_sq=mu.pow(2)
        sigma_sq = F.conv2d(img*img, self.window, padding = self.window_size//2, groups = self.channel) - mu_sq
        B,C,W,H=sigma_sq.shape
        sigma_sq = torch.sqrt(sigma_sq)
        sigma_sq=torch.flatten(sigma_sq, start_dim=1)
        noise_map = self.softmax(sigma_sq)
        noise_map=torch.reshape(noise_map,[B,C,W,H])
        return noise_map

class NCMSE(nn.Module):
    # Noise Concious MSE Loss referred from https://github.com/reach2sbera/ldct_nonlocal
    """
    The Noise Conscious MSE Loss 
    """
    def __init__(self):
        super(NCMSE, self).__init__()
        self.std=STD()
    def forward(self, out_image, gt_image, org_image):
        _, C, H, W = out_image.size()
        N = C*H*W
        loss = -torch.mean(torch.mul(self.std(org_image - gt_image), torch.pow(out_image - gt_image, 2))) / float(N) 
        return loss

class MPL(torch.nn.Module):
    """
    The Multi Perceptual Loss Function
    """
    def __init__(self):
        super(MPL, self).__init__()
        self.model = models.vgg19(pretrained=True)
        self.model.to(torch.device('cuda' if cuda_is_present else 'cpu'))
    def forward(self, target, prediction):
        perceptual_loss = 0
        vgg16_layers = [3, 8, 17, 26, 35] # layers: 3, 8, 17, 26, 35
        for layer in vgg16_layers:
            feature_target = get_feature_layer_vgg16(target, layer, self.model)
            feature_prediction = get_feature_layer_vgg16(prediction, layer, self.model)
            _, C, H, W = feature_target.size()
            feature_count = C*W*H
            feature_difference = feature_target - feature_prediction
            feature_loss = feature_difference.norm(p=2) / float(feature_count)
            perceptual_loss += feature_loss
        return perceptual_loss

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1

        self.window = create_window(window_size, self.channel)
        self.window.to(torch.device('cuda' if cuda_is_present else 'cpu'))
    def forward(self, y, pred):
        return compute_SSIM(y, pred, self.window_size, self.channel, self.size_average)

class DLoss(torch.nn.Module):
    """
    The loss for discriminator
    """
    def __init__(self):
        super(DLoss, self).__init__()
        # self.weight = weight
    def forward(self, Dy, Dg):
        return -torch.mean(Dy) + torch.mean(Dg)

class GLoss(torch.nn.Module):
    """
    The loss for generator
    """
    def __init__(self):
        super(GLoss, self).__init__()
    def forward(self, Dg, pred, y):
        ADVERSARIAL_LOSS_FACTOR, PIXEL_LOSS_FACTOR = 0.5, 1.0
        loss = ADVERSARIAL_LOSS_FACTOR * -torch.mean(Dg) #+ PIXEL_LOSS_FACTOR * mse_loss(y,pred)
        return loss