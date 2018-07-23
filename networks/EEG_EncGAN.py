import utils, torch, time, os, pickle, imageio, math
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from torch.autograd import Variable, grad
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from utils import Flatten
from spectral_normalization import SpectralNorm
import pdb

class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()
		self.input_dim = 3
		self.input_height = 64
		self.input_width = 64
		self.output_dim = 50

		self.conv = nn.Sequential(
			nn.Conv2d(self.input_dim, 64, 3, 4, 2, bias=True),
			nn.BatchNorm2d(64),
			#nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),

			nn.Conv2d(64, 128, 4, 2, 1, bias=True),
			nn.BatchNorm2d(128),
			#nn.InstanceNorm2d(128, affine=True),
			nn.ReLU(),

			nn.Conv2d(128, 256, 4, 2, 1, bias=True),
			nn.BatchNorm2d(256),
			#nn.InstanceNorm2d(256, affine=True),
			nn.ReLU(),

			nn.Conv2d(256, 512, 4, 2, 1, bias=True),
			nn.BatchNorm2d(512),
			#nn.InstanceNorm2d(512, affine=True),
			nn.ReLU(),

			nn.Conv2d(512, self.output_dim, 4, 2, 1, bias=True),
			nn.Sigmoid(),
		)

		utils.initialize_weights(self)

	def forward(self, input):
		x = self.conv(input).squeeze(3).squeeze(2)
		return x

class Decoder(nn.Module):
	def __init__(self):
		super(Decoder, self).__init__()
		self.input_dim = 128
		self.output_dim = 3 

		self.fc = nn.Sequential(
			nn.Linear(150, self.input_dim)
		)

		self.deconv = nn.Sequential(
			
			#4
			nn.Conv2d(self.input_dim, 512, 4, 1, 3, bias=False),
			nn.BatchNorm2d(512),
			nn.ReLU(),

			#8
			nn.Upsample(scale_factor=2, mode='nearest'),
			nn.Conv2d(512, 256, 3, 1, 1, bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(),

			#16
			nn.Upsample(scale_factor=2, mode='nearest'),
			nn.Conv2d(256, 128, 3, 1, 1, bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU(),

			#32
			nn.Upsample(scale_factor=2, mode='nearest'),
			nn.Conv2d(128, 64, 3, 1, 1, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(),

			#64
			nn.Upsample(scale_factor=2, mode='nearest'),
			nn.Conv2d(64, self.output_dim, 3, 1, 1, bias=False),
			nn.Sigmoid(),
		
		)
		utils.initialize_weights(self)

	def forward(self, z, spc):
		feature = torch.cat((z, spc),1)
		x = self.fc(feature)
		x = self.deconv(x.view(-1, self.input_dim, 1, 1))
		return x


class Generator(nn.Module):
	def __init__(self, num_cls):
		super(Generator, self).__init__()
		self.num_cls = num_cls

		self.Enc = Encoder()
		self.Dec = Decoder()

		utils.initialize_weights(self)

	def forward(self, spc, z):
		spc_ = self.Enc(spc)
		result = self.Dec(z, spc_)

		return result

class Discriminator(nn.Module):
	def __init__(self, num_cls):
		super(Discriminator, self).__init__()
		self.input_dim = 3
		self.num_cls = num_cls

		self.conv = nn.Sequential(
			#64->32
			nn.Conv2d(self.input_dim, 32, 4, 2, 1, bias=False),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(),

			#32->16
			nn.Conv2d(32, 64, 4, 2, 1, bias=False),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.2),

			#16->8
			nn.Conv2d(64, 128, 4, 2, 1, bias=False),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2),

			#8->4
			nn.Conv2d(128, 256, 4, 2, 1, bias=False),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2),
		)

		self.convCls = nn.Sequential(
			nn.Conv2d(256, self.num_cls, 4, bias=False)
		)

		self.convGAN = nn.Sequential(
			nn.Conv2d(256, 1, 4, bias=False),
			nn.Sigmoid()
		)


		utils.initialize_weights(self)

	def forward(self, y):
		feature = self.conv(y)
		fGAN = self.convGAN(feature).squeeze(3).squeeze(2)
		fcls = self.convCls(feature).squeeze(3).squeeze(2)

		return fGAN, fcls


class EEG_EncGAN(object):
	def __init__(self, args):
		#parameters
		self.batch_size = args.batch_size
		self.epoch = args.epoch
		self.save_dir = args.save_dir
		self.result_dir = args.result_dir
		self.dataset = 'EEG_ImageNet'#args.dataset
		self.dataroot_Img_dir = '../../eegImagenet/mindbigdata-imagenet-in-v1.0/MindBigData-Imagenet-v1.0-Imgs'
		self.dataroot_EEG_dir = '../../eegImagenet/mindbigdata-imagenet-in-v1.0/MindBigData-Imagenet'
		self.model_name = args.gan_type + args.comment
		self.sample_num = args.sample_num
		self.gpu_mode = args.gpu_mode
		self.num_workers = args.num_workers
		self.beta1 = args.beta1
		self.beta2 = args.beta2
		self.lrG = args.lrG
		self.lrD = args.lrD
		self.lrE = args.lrD
		self.type = 'train'
		self.lambda_ = 0.25
		self.n_critic = args.n_critic
		self.d_trick = args.d_trick
		self.use_recon = True

		self.enc_dim = 100
#		self.num_cls = 100

		#load dataset
		self.data_loader = DataLoader(utils.EEG_ImageNet(root_dir = self.dataroot_EEG_dir,transform=transforms.Compose([transforms.Scale(64),transforms.RandomCrop(64),transforms.ToTensor()]),_type = self.type), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
			
		#self.num_cls = self.data_loader.dataset.num_cls
		self.num_cls = len(self.data_loader.dataset.cls_map)

		self.G = Generator(self.num_cls)
		self.D = Discriminator(self.num_cls)

		self.G_optimizer = optim.Adam(self.G.parameters(), lr=self.lrG, betas=(self.beta1, self.beta2))
		self.D_optimizer = optim.Adam(self.D.parameters(), lr=self.lrD, betas=(self.beta1, self.beta2))

		if self.gpu_mode:
			self.G = self.G.cuda()
			self.D = self.D.cuda()
			self.CE_loss = nn.CrossEntropyLoss().cuda()
			self.L1_loss = nn.L1Loss().cuda()
			self.BCE_loss = nn.BCELoss().cuda()
		else:
			self.CE_loss = nn.CrossEntropyLoss()
			self.L1_loss = nn.L1Loss()
			self.BEC_loss = nn.BECLoss()
	

	def train(self):
		self.train_hist = {}
		self.train_hist['G_loss'] = []
		self.train_hist['D_loss'] = []
		self.train_hist['per_epoch_time']=[]
		self.train_hist['total_time']=[]

		if self.gpu_mode:
			self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1).cuda()), Variable(torch.zeros(self.batch_size, 1).cuda())
		else:
			self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1)), Variable(torch.zeros(self.batch_size, 1))

		#train
		self.D.train()
		start_time = time.time()
		for epoch in range(self.epoch):
			epoch_start_time = time.time()
			self.G.train()
			for iB, (eeg, x, spc, class_label) in enumerate(self.data_loader):
				if iB == self.data_loader.dataset.__len__() // self.batch_size:
					break

				#--Make Latent Space--#
				z_ = torch.rand(self.batch_size, self.enc_dim)
				#z_ = torch.FloatTensor(self.batch_size, self.enc_dim).normal_(0.0, 1.0)

				if self.gpu_mode:
					x_, z_, class_label_, eeg_, spc_ = Variable(x.cuda()), Variable(z_.cuda()), Variable(class_label.cuda()), Variable(eeg.cuda()), Variable(spc.cuda())
				else:
					x_, z_, class_label_, eeg_, spc_ = Variable(x), Variable(z_), Variable(class_label), Variable(eeg), Variable(spc)


				#----Update D_network----#
				self.D_optimizer.zero_grad()
				D_real, C_real = self.D(x_)
				D_real_loss = self.BCE_loss(D_real, self.y_real_)
				C_real_loss = self.CE_loss(C_real, class_label_)
				
				G_ = self.G(spc_, z_)
				D_fake, C_fake = self.D(G_)
				D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)
				C_fake_loss = self.CE_loss(C_fake, class_label_)
				
				D_ganloss = D_real_loss + D_fake_loss
				D_clsloss = C_real_loss + C_fake_loss

				#gradient penalry
				if self.gpu_mode:
					alpha = torch.rand(x_.size()).cuda()
				else:
					alpha = torch.rand(x_.size())

				x_hat = Variable(alpha * x_.data + (1-alpha)*G_.data, requires_grad=True)

				pred_hat, class_hat = self.D(x_hat)
				if self.gpu_mode:
					gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).cuda(), create_graph=True, retain_graph=True, only_inputs=True)[0]
				else:
					gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()), create_output=True, retain_graph = True, only_inputs=True)[0]

				gradient_penalty = self.lambda_ *((gradients.view(gradients.size()[0], -1).norm(2,1) -1)**2).mean()

				D_loss = D_ganloss + D_clsloss + gradient_penalty
				self.train_hist['D_loss'].append(D_loss.data[0])
				
				num_correct_real = torch.sum(D_real > 0.5)
				num_correct_fake = torch.sum(D_fake < 0.5)

				D_acc = float(num_correct_real.data[0] + num_correct_fake.data[0]) / (self.batch_size*2)
				D_loss.backward()
				if self.d_trick:
					if (D_acc < 0.8):
						self.D_optimizer.step()
				else:
					self.D_optimizer.step()

				#----Update G_network----#
				for iG in range(self.n_critic):
					self.G_optimizer.zero_grad()
					G_ = self.G(spc_, z_)
					D_fake, C_fake = self.D(G_)
					G_fake_loss = self.BCE_loss(D_fake, self.y_real_)
					G_cls_loss = self.CE_loss(C_fake, class_label_)

					if self.use_recon:
						G_recon_loss = self.L1_loss(G_, x_)
						G_loss = G_fake_loss + G_cls_loss + G_recon_loss*80
					else:
						G_loss = G_fake_loss + G_cls_loss
					if iG == (self.n_critic-1):
						self.train_hist['G_loss'].append(G_loss.data[0])
					G_loss.backward()
					self.G_optimizer.step()
				#---check train result ----#
				if(iB % 100 == 0):
					print('[E%03d]'%(epoch)+'D_loss : ',D_loss.data[0],' = ', D_ganloss.data[0],' + ',D_clsloss.data[0], '  G_loss : ', G_fake_loss.data[0],' + ' , G_cls_loss.data[0], 'D_acc :', D_acc)
					self.visualize_results(epoch, eeg_, spc_, z_, x_, iB)
			#---check train result ----#
			self.train_hist['per_epoch_time'].append(time.time()-epoch_start_time)
			utils.loss_plot(self.train_hist, os.path.join(self.result_dir, self.dataset, self.model_name), self.model_name)
		
		print("Training finish!... save training results")
		self.save()


	def visualize_results(self, epoch, eeg, spc, z, y, iB, fix=True):
		self.G.eval()
		if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
			os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

		tot_num_samples = min(self.sample_num, self.batch_size)
		image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

		if fix:
			""" fixed noise """
			samples = self.G(spc, z)
		
		if self.gpu_mode:
			samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
			gt = y.cpu().data.numpy().transpose(0, 2, 3, 1)
		else:
			samples = samples.data.numpy().transpose(0, 2, 3, 1)
			gt = y.data.numpy().transpose(0, 2, 3, 1)

		utils.save_images(samples[:image_frame_dim*image_frame_dim,:,:,:], [image_frame_dim, image_frame_dim], self.result_dir+'/'+self.dataset+'/'+self.model_name+'/'+self.model_name+'_epoch%03d'%epoch+'_I%03d'%iB+'.png')
		utils.save_images(gt[:image_frame_dim*image_frame_dim,:,:,:], [image_frame_dim, image_frame_dim], self.result_dir+'/'+self.dataset+'/'+self.model_name+'/'+self.model_name+'gt_epoch%03d'%epoch+'_I%03d'%iB+'.png')


	def save(self):
		save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
		torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

		with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
			pickle.dump(self.train_hist, f)

	def load(self):
		save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

		self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
		self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))


