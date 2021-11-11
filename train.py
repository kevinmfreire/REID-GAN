import os
import argparse
#import imageio
import numpy as np
import matplotlib.pyplot as plt

# Importing torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# For MNIST dataset and visualization
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
from loss import *
from networks import *
from loader import get_loader


parser= argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--load_mode', type=int, default=1)
parser.add_argument('--data_path', type=str, default='./patient')
parser.add_argument('--saved_path', type=str, default='./patient/data/npy_img/')
parser.add_argument('--save_path', type=str, default='./model/')
parser.add_argument('--test_patient', type=str, default='L064')

parser.add_argument('--transform', type=bool, default=False)
# if patch training, batch size is (--patch_n * --batch_size)
parser.add_argument('--patch_n', type=int, default=None)		# default = 4
parser.add_argument('--patch_size', type=int, default=None)	# default = 100
parser.add_argument('--batch_size', type=int, default=5)	# default = 5

parser.add_argument('--lr', type=float, default=0.0002) # Defailt = 1e-3

parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--num_workers', type=int, default=7)
parser.add_argument('--load_chkpt', type=bool, default=False)

args = parser.parse_args()

data_loader = get_loader(mode=args.mode,
                             load_mode=args.load_mode,
                             saved_path=args.saved_path,
                             test_patient=args.test_patient,
                             patch_n=(args.patch_n if args.mode=='train' else None),
                             patch_size=(args.patch_size if args.mode=='train' else None),
                             transform=args.transform,
                             batch_size=(args.batch_size if args.mode=='train' else 1),
                             num_workers=args.num_workers)

# Check CUDA's presence
cuda_is_present = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda_is_present else torch.FloatTensor

def to_cuda(data):
    	return data.cuda() if cuda_is_present else data

if args.load_chkpt:
	print('Loading Chekpoint')
	whole_model = torch.load(args.save_path+ 'latest_ckpt.pth.tar')
	netG_state_dict,optG_state_dict = whole_model['netG_state_dict'], whole_model['optG_state_dict']
	netD_state_dict,optD_state_dict = whole_model['netD_state_dict'], whole_model['optD_state_dict']
	g_net = GNet()
	g_net = to_cuda(g_net)
	d_net = DNet()
	d_net = to_cuda(d_net)
	optimizer_generator = torch.optim.Adam(g_net.parameters())
	optimizer_discriminator= torch.optim.Adam(d_net.parameters())
	d_net.load_state_dict(netD_state_dict)
	g_net.load_state_dict(netG_state_dict)
	optimizer_generator.load_state_dict(optG_state_dict)
	optimizer_discriminator.load_state_dict(optD_state_dict)
	cur_epoch = whole_model['epoch']
	total_iters = whole_model['total_iters']
	lr = whole_model['lr']
	# netG = torch.nn.DataParallel(netG, device_ids=[0, 1])
	# netD = torch.nn.DataParallel(netD, device_ids=[0, 1])
	print('Current Epoch:{}, Total Iters: {}, Learning rate: {}, Batch size: {}'.format(cur_epoch, total_iters, lr, args.batch_size))
else:
	print('Training model from scrath')
	g_net = GNet()
	g_net = to_cuda(g_net)
	d_net = DNet()
	d_net = to_cuda(d_net)
	optimizer_generator = torch.optim.Adam(g_net.parameters(), lr=args.lr)
	optimizer_discriminator = torch.optim.Adam(d_net.parameters(), lr=4*args.lr)
	cur_epoch = 0
	total_iters = 0
	lr=args.lr

# Losses
Dloss = DLoss()
# Dloss.cuda()
Gloss = GLoss()
Gloss = to_cuda(Gloss)

losses = []
torch.autograd.set_detect_anomaly(True)
for epoch in range(cur_epoch, args.num_epochs):
	g_net.train()
	for i, (x, y) in enumerate(data_loader):
	# for i in range(20):
		total_iters += 1
		shape_ = x.shape[-1]

		# add 1 channel
		x = x.unsqueeze(0).float()
		y = y.unsqueeze(0).float()

		if args.patch_size:     #patch training
			x = x.view(-1, 1, args.patch_size, args.patch_size)
			y = y.view(-1, 1, args.patch_size, args.patch_size)

		if args.batch_size:     #batch training
			x = x.view(-1, 1, shape_, shape_)
			y = y.view(-1, 1, shape_, shape_)

		x = to_cuda(x)
		y = to_cuda(y)

		# NEW PREDICTIONS
		pred = g_net(x)

		# TRAINING DISCRIMINATOR
		optimizer_discriminator.zero_grad()
		d_net.zero_grad()
		Dy = d_net(y)
		Dg = d_net(pred)
		dloss = Dloss(Dy,Dg)
		dloss.backward(retain_graph=True)
		optimizer_discriminator.step()

		# TRAINING GENERATOR
		optimizer_generator.zero_grad()
		g_net.zero_grad()
		gloss = Gloss(Dg, pred, y)
		gloss.backward()
		optimizer_generator.step()

		print(f'losses generator : {gloss.item()}, discriminator : {dloss.item()}')
		# print(f'losses generator : {generator_loss}, discriminator : {discriminator_loss}')

	losses.append((gloss.item(), dloss.item()))
	# losses.append((generator_loss.item(), discriminator_loss.item()))

	# Saving model after every epoch
	if not os.path.exists(args.save_path):
		os.makedirs(args.save_path)
		print('Create path : {}'.format(args.save_path))
	print('Saving model to: ' + args.save_path)
	saved_model = {
		'epoch': epoch ,
		'netG_state_dict': g_net.state_dict(),
		'optG_state_dict': optimizer_generator.state_dict(),
		'netD_state_dict': d_net.state_dict(),
		'optD_state_dict': optimizer_discriminator.state_dict(),
		'lr': lr,
		'total_iters': total_iters
	}
	if not os.path.exists(args.save_path):
		os.makedirs(args.save_path)
		print('Create path : {}'.format(args.save_path))
	# torch.save(saved_model, '{}iter_{}_ckpt.pth.tar'.format(args.save_path, total_iters))
	torch.save(saved_model, '{}iter_{}_ckpt.pth.tar'.format(args.save_path, total_iters))
	torch.save(saved_model, '{}latest_ckpt.pth.tar'.format(args.save_path))