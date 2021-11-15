
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

MSEloss = torch.nn.MSELoss()
MSEloss = to_cuda(MSEloss)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 3.5).unsqueeze(1)
    _2D_window =_1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return  window # window.cuda()

def get_feature_layer_vgg16(image):
    image = torch.cat([image,image,image],1)
    vgg16 = models.vgg16(pretrained=True)
    vgg16 = to_cuda(vgg16)
    return_layers = {'8': 'out_layer8'}
    # THIS PRINTS UP TO THE OUTPUT OF THE SECOND CONV2_2 LAYERS FOR FEATURE EXTRACTION
    # (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # (1): ReLU(inplace=True)
    # (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # (3): ReLU(inplace=True)
    # (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    # (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # (6): ReLU(inplace=True)
    # (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # (8): ReLU(inplace=True)
    output_feature = IntermediateLayerGetter(vgg16.features, return_layers=return_layers)
    image_feature = output_feature(image)
    return image_feature['out_layer8']

def get_feature_loss(target,prediction):
    feature_transformed_target = get_feature_layer_vgg16(target)
    feature_transformed_prediction = get_feature_layer_vgg16(prediction)
    feature_count = feature_transformed_target.shape[-1]
    feature_difference = feature_transformed_target-feature_transformed_prediction
    feature_loss = torch.sum(feature_difference.pow(2))
    # feature_loss = torch.sum(torch.square(feature_transformed_target-feature_transformed_prediction))
    feature_loss = feature_loss/float(feature_count)
    return feature_loss

def get_smooth_loss(image):
    _, _ , image_height, image_width = image.size()
    horizontal_normal = image[:, :, :, 0:image_width-1]
    horizontal_one_right = image[:, :, :, 1:image_width]
    vertical_normal = image[:, :, 0:image_height-1, :]
    vertical_one_right = image[:, :, 1:image_height, :]
    # smooth_loss = torch.nn.MSELoss(horizontal_normal,horizontal_one_right) + torch.nn.MSELoss(vertical_normal,vertical_one_right)
    smooth_loss = torch.pow(horizontal_normal-horizontal_one_right, 2).sum() / 2.0 + torch.pow(vertical_normal - vertical_one_right, 2).sum()/2.0
    return smooth_loss

class DLoss(torch.nn.Module):
    """
    The loss for discriminator
    """
    def __init__(self):
        super(DLoss, self).__init__()
        # self.weight = weight

    def forward(self, Dy, Dg):
        return -torch.mean(torch.log(Dy) + torch.log1p(-Dg))
        # return - self.weight * torch.mean(neg)

class GLoss(torch.nn.Module):
    """
    The loss for generator
    """
    def __init__(self,ADVERSARIAL_LOSS_FACTOR=0.5, PIXEL_LOSS_FACTOR=1.0, FEATURE_LOSS_FACTOR=1.0, SMOOTH_LOSS_FACTOR=0.0001):
        super(GLoss, self).__init__()
        self.ADVERSARIAL_LOSS_FACTOR = ADVERSARIAL_LOSS_FACTOR
        self.PIXEL_LOSS_FACTOR = PIXEL_LOSS_FACTOR
        self.FEATURE_LOSS_FACTOR = FEATURE_LOSS_FACTOR
        self.SMOOTH_LOSS_FACTOR = SMOOTH_LOSS_FACTOR

    def forward(self, Dg, pred, y):
        loss = self.ADVERSARIAL_LOSS_FACTOR * -torch.mean(torch.log(Dg)) + self.PIXEL_LOSS_FACTOR * MSEloss(y,pred) + \
			self.FEATURE_LOSS_FACTOR * get_feature_loss(y,pred) + self.SMOOTH_LOSS_FACTOR * get_smooth_loss(pred)
        return loss