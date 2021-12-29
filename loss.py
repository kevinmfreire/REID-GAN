import torch.nn as nn
import torch
import torch.nn.functional as F
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
    _1D_window = gaussian(window_size, 3.5).unsqueeze(1)
    _2D_window =_1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return  window # window.cuda()

def get_pixel_loss(target, prediction):
    B, C, H, W = target.size()
    N = B*C*H*W
    pixel_difference = target - prediction
    pixel_loss = pixel_difference.norm(p=2) / float(N)
    # loss = nn.MSELoss()
    # pixel_loss = loss(prediction, target)
    return pixel_loss

# OBTAIN VGG16 PRETRAINED MODEL EXCLUDING FULLY CONNECTED LAYERS
def get_feature_layer_vgg16(image):
    image = torch.cat([image,image,image],1)
    vgg16 = models.vgg16(pretrained=True)
    vgg16 = to_cuda(vgg16)
    return_layers = {'29': 'out_layer29'}
    output_feature = IntermediateLayerGetter(vgg16.features, return_layers=return_layers)
    image_feature = output_feature(image)
    return image_feature['out_layer29']

def get_feature_loss(target,prediction):
    feature_transformed_target = get_feature_layer_vgg16(target)
    feature_transformed_prediction = get_feature_layer_vgg16(prediction)
    _, C, H, W = feature_transformed_target.size()
    feature_count = C*W*H
    feature_difference = feature_transformed_prediction - feature_transformed_target
    feature_loss = feature_difference.norm(p=2) / float(feature_count)
    # loss = nn.MSELoss()
    # feature_loss = loss(feature_transformed_prediction, feature_transformed_target)
    return feature_loss

def get_smooth_loss(image):
    _, _ , image_height, image_width = image.size()
    horizontal_normal = image[:, :, :, 0:image_width-1]
    horizontal_one_right = image[:, :, :, 1:image_width]
    vertical_normal = image[:, :, 0:image_height-1, :]
    vertical_one_right = image[:, :, 1:image_height, :]
    # smooth_loss = torch.sum(torch.pow(horizontal_normal-horizontal_one_right, 2)) / 2.0 + torch.sum(torch.pow(vertical_normal - vertical_one_right, 2)) / 2.0
    smooth_loss = get_pixel_loss(horizontal_normal, horizontal_one_right) + get_pixel_loss(vertical_normal, vertical_one_right) 
    return smooth_loss

class DLoss(torch.nn.Module):
    """
    The loss for discriminator
    """
    def __init__(self):
        super(DLoss, self).__init__()
        # self.weight = weight

    def forward(self, Dy, Dg):
        # return -torch.mean(Dy) + torch.mean(Dg)
        return -(torch.log(Dy) + torch.log(1.-Dg))

class GLoss(torch.nn.Module):
    """
    The loss for generator
    """
    def __init__(self):
        super(GLoss, self).__init__()

    def forward(self, Dg, pred, y):
        ADVERSARIAL_LOSS_FACTOR, PIXEL_LOSS_FACTOR, FEATURE_LOSS_FACTOR = 0.5, 0.1, 0.1
        loss = ADVERSARIAL_LOSS_FACTOR * -torch.log(Dg) + PIXEL_LOSS_FACTOR * get_pixel_loss(y,pred) + \
			FEATURE_LOSS_FACTOR * get_feature_loss(y,pred)
        return loss