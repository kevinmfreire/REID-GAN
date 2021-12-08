import os
import argparse
#import imageio
import numpy as np
import matplotlib.pyplot as plt
import time

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
from tqdm import tqdm


parser= argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--load_mode', type=int, default=1)
parser.add_argument('--data_path', type=str, default='./patient')
parser.add_argument('--saved_path', type=str, default='./patient/data/npy_img/')
parser.add_argument('--save_path', type=str, default='./normalized_model/')
parser.add_argument('--test_patient', type=str, default='L064')

parser.add_argument('--save_iters', type=int, default=10)
parser.add_argument('--print_iters', type=int, default=50)
parser.add_argument('--decay_iters', type=int, default=6000)
parser.add_argument('--gan_alt', type=int, default=2)

parser.add_argument('--transform', type=bool, default=False)
# if patch training, batch size is (--patch_n * --batch_size)
parser.add_argument('--patch_n', type=int, default=10)		# default = 4
parser.add_argument('--patch_size', type=int, default=120)	# default = 100
parser.add_argument('--batch_size', type=int, default=16)	# default = 5
parser.add_argument('--image_size', type=int, default=512)

parser.add_argument('--lr', type=float, default=1e-4) # Defailt = 1e-3
parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--load_chkpt', type=bool, default=False)

parser.add_argument('--norm_range_min', type=float, default=-1024.0)
parser.add_argument('--norm_range_max', type=float, default=3071.0)

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

def denormalize_(image):
    image = image * (args.norm_range_max - args.norm_range_min) + args.norm_range_min
    return image

def normalize_(image):
    image = (image - args.norm_range_min) / (args.norm_range_max - args.norm_range_min)
    return image

image_size = args.image_size if args.patch_size == None else args.patch_size

if args.load_chkpt:
	print('Loading Chekpoint')
	whole_model = torch.load(args.save_path+ 'latest_ckpt.pth.tar')
	netG_state_dict,optG_state_dict = whole_model['netG_state_dict'], whole_model['optG_state_dict']
	netD_state_dict,optD_state_dict = whole_model['netD_state_dict'], whole_model['optD_state_dict']
	g_net = GNet(image_size)
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
	# g_net = torch.nn.DataParallel(g_net, device_ids=[0, 1])
	# d_net = torch.nn.DataParallel(d_net, device_ids=[0, 1])
	print('Current Epoch:{}, Total Iters: {}, Learning rate: {}, Batch size: {}'.format(cur_epoch, total_iters, lr, args.batch_size))
else:
	print('Training model from scrath')
	g_net = GNet(image_size)
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
Dloss = to_cuda(Dloss)
Gloss = GLoss()
Gloss = to_cuda(Gloss)

losses = []
train_dis = True
torch.autograd.set_detect_anomaly(True)
gen_count = 0
start_time = time.time()
tq_epoch = tqdm(range(cur_epoch, args.num_epochs),position=1, leave=True, desc='Epochs')
print("Training Discriminator")
for epoch in tq_epoch:

	# Initializing sum of losses for discriminator and generator
	gloss_sum, dloss_sum, count = 0, 0, 0
	
	# Alteranating training between discriminator and generator
	train_gen = gen_count >= args.gan_alt
	if not train_gen:
		print(f"Training Discriminator {gen_count+1}")
	else:
		print(f"Training Generator  {gen_count - 1}")
	g_net.train(train_gen)
	d_net.train(not train_gen)
	gen_count += 1
	if gen_count == args.gan_alt*5 + args.gan_alt:
		gen_count = 0

	data_tqdm = tqdm(data_loader, position=0, leave=True, desc='Iters')
	for i, (x, y) in enumerate(data_tqdm):
		total_iters += 1
		count += 1
		shape_ = x.shape[-1]

		# add 1 channel
		x = x.unsqueeze(0).float()
		y = y.unsqueeze(0).float()

		# If patch training
		if args.patch_size:
			x = x.view(-1, 1, args.patch_size, args.patch_size)
			y = y.view(-1, 1, args.patch_size, args.patch_size)

		# If batch training without any patch size
		if args.batch_size and args.patch_size == None:
			x = x.view(-1, 1, shape_, shape_)
			y = y.view(-1, 1, shape_, shape_)

		y = to_cuda(y)
		x = to_cuda(x)
		x = normalize_(x)

		# Predictions
		pred = g_net(x)
		pred = denormalize_(pred)

		# Training discriminator
		optimizer_discriminator.zero_grad()
		d_net.zero_grad()
		Dy = d_net(y)
		Dg = d_net(pred)
		dloss = Dloss(Dy,Dg)
		
		dloss.backward(retain_graph=True)
		optimizer_discriminator.step()

		# Training generator
		optimizer_generator.zero_grad()
		g_net.zero_grad()
		gloss = Gloss(Dg, pred, y)
		gloss.backward()
		optimizer_generator.step()
		dloss_sum += dloss.item()
		gloss_sum += gloss.item()
		

		# Print progress after every 50 iterations
		# if total_iters % args.print_iters == 0:
		# 	print("STEP [{}], EPOCH [{}/{}], ITER [{}/{}] \nG_LOSS: {:.8f}, D_LOSS: {:.14f}, TIME: {:.1f}s".format(total_iters, epoch, 
		# 																								args.num_epochs, i+1, 
		# 																								len(data_loader), gloss.item(), dloss.item()
		# 																								,time.time() - start_time))
		data_tqdm.set_postfix({'ITER': i+1, 'G_LOSS': '{:.5f}'.format(gloss.item()), 'D_LOSS': '{:.8f}'.format(dloss.item())})
		if total_iters % args.decay_iters == 0:
			lr = lr * 0.5
			for param_group in optimizer_generator.param_groups:
				param_group['lr'] = lr
			for param_group in optimizer_discriminator.param_groups:
				param_group['lr'] = 4*lr

		# Saving model after every 10 iterations
		if total_iters % args.save_iters == 0:
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
			# torch.save(saved_model, '{}iter_{}_ckpt.pth.tar'.format(args.save_path, total_iters))
			torch.save(saved_model, '{}latest_ckpt.pth.tar'.format(args.save_path))
			# save_model(saved_model, '{}latest_ckpt.pth.tar'.format(args.save_path))
			cmd = 'cp {}latest_ckpt.pth.tar /gdrive/MyDrive/normalized_model/'.format(args.save_path)
			os.system(cmd)
	
	# Calculating average loss
	avg_dloss = dloss_sum/float(count)
	avg_gloss = gloss_sum/float(count)
	
	# print("STEP [{}], EPOCH [{}/{}], ITER [{}/{}] \nAVG_G_LOSS: {:.8f}, AVG_D_LOSS: {:.14f}, TIME: {:.1f}s".format(total_iters, epoch, 
	# 																									args.num_epochs, i+1, 
	# 																									len(data_loader), avg_gloss, avg_dloss
	# 																									,time.time() - start_time))
	tq_epoch.set_postfix({'STEP': total_iters,'AVG_G_LOSS': '{:.5f}'.format(avg_gloss), 'AVG_D_LOSS': '{:.8f}'.format(avg_dloss)})
	# Saving model after every 10 epoch
	if epoch % 10 == 0:
		cmd1 = 'cp {}latest_ckpt.pth.tar /gdrive/MyDrive/normalized_model/epoch_{}_ckpt.pth.tar'.format(args.save_path, epoch)
		cmd2 = 'cp {}latest_ckpt.pth.tar /gdrive/MyDrive/normalized_model/'.format(args.save_path)
		os.system(cmd1)
		os.system(cmd2)
	losses.append((gloss.item(), dloss.item()))