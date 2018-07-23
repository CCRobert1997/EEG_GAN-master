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

class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()
		self.input_dim = 128
		self.output_dim = 3

		self.fc = nn.Sequential(
			nn.Linear(140, self.input_dim),
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
			nn.Conv2d(64, 32, 3, 1, 1, bias=False),
			nn.BatchNorm2d(32),
			nn.ReLU(),

			#128
			nn.Upsample(scale_factor=2, mode='nearest'),
			nn.Conv2d(32, self.output_dim, 3, 1, 1, bias=False),
			nn.Sigmoid()		
		)

		utils.initialize_weights(self)

	def forward(self, z, gru):
		feature = torch.cat((z, gru), 1)
		x = self.fc(feature)
		x = self.deconv(x.view(-1, self.input_dim, 1, 1))
		return x

class Discriminator(nn.Module):
	def __init__(self, num_cls):
		super(Discriminator, self).__init__()
		self.input_dim = 3
		self.num_cls = num_cls
		
		self.conv = nn.Sequential(
			#128->64
			nn.Conv2d(self.input_dim+self.num_cls, 32, 4, 2, 1, bias=False),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(0.2),

			#64->32
			nn.Conv2d(32, 64, 4, 2, 1, bias=False),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.2),

			#32->16
			nn.Conv2d(64, 128, 4, 2, 1, bias=False),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2),

			#16->8
			nn.Conv2d(128, 256, 4, 2, 1, bias=False),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2),

			#8->4
			nn.Conv2d(256, 512, 4, 2, 1, bias=False),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.2),
		
		)

		self.convGAN = nn.Sequential(
			nn.Conv2d(512, 1, 4, bias=False),
			nn.Sigmoid()
		)

		self.convCls = nn.Sequential(
			nn.Conv2d(512, self.num_cls, 4, bias=False),
		)

		utils.initialize_weights(self)

	def forward(self, y, label):
		feature = torch.cat([y,label],1)
		feature = self.conv(feature)
		fGAN = self.convGAN(feature).squeeze(3).squeeze(2)
		#fcls = self.convCls(feature).squeeze(3).squeeze(2)

		return fGAN#, fcls


class GRU_Encoder(nn.Module):
	def __init__(self, num_cls):
		super(GRU_Encoder, self).__init__()
		self.hidden_dim = 64
		self.input_dim = 128
		self.output_dim = 128 # class_num or feature dimension(for concat)
	
		self.GRU = nn.GRU(self.input_dim, self.hidden_dim, num_layers = 2,  batch_first = True, dropout=0.5)
		self.fc = nn.Sequential(
			nn.Linear(self.hidden_dim , self.output_dim),
			nn.ReLU(),
			#nn.Sigmoid()
		)

	def forward(self, feature):

		feature = feature.transpose(1,2)#dimension change : batch x time x dimension	
		x, hidden = self.GRU(feature)
		x = x.select(1, x.size(1)-1).contiguous()
		x = x.view(-1, self.hidden_dim)
		result = self.fc(x)
		return result


class Generator_Union(nn.Module):
	def __init__(self, num_cls):
		super(Generator_Union, self).__init__()
		self.num_cls = num_cls

		self.GRU = GRU_Encoder(self.num_cls)
		self.Enc = Generator()

	def forward(self, eeg_, z_):
		eeg_ = self.GRU(eeg_)
		x = self.Enc(z_, eeg_)
		return x


class EEG_GRU_GAN(object):
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
		self.use_recon = args.use_recon
		self.use_gp = args.use_gp
		self.mini = args.num_cls
		self.enc_dim = 100
		self.resl = 128


		#load dataset
		'''
		self.data_loader = DataLoader(utils.EEG(root_dir = self.dataroot_EEG_dir, transform=transforms.Compose([transforms.Scale(64),transforms.RandomCrop(64),transforms.ToTensor()]),_type = self.type), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

		self.test_loader = DataLoader(utils.EEG(root_dir = self.dataroot_EEG_dir, transform=None, _type = 'test'), batch_size = self.batch_size, shuffle=True, num_workers=self.num_workers)
		'''
		self.train_loader = DataLoader(utils.EEG_pytorch(root_dir = './', transform = transforms.Compose([transforms.Scale(int(self.resl*1.5)), transforms.RandomCrop(self.resl), transforms.ToTensor()]) , _type = 'train'), batch_size = self.batch_size , shuffle=True, num_workers = self.num_workers)

		self.test_loader = DataLoader(utils.EEG_pytorch(root_dir = './', transform = None, _type = 'test'), batch_size = self.batch_size, shuffle=True, num_workers = self.num_workers)


		#self.num_cls = self.data_loader.dataset.num_cls
		self.num_cls = 40#len(self.data_loader.dataset.cls_map)

#		self.GRU = GRU_Encoder(self.num_cls)
#		self.GRU_optimizer = optim.Adam(self.GRU.parameters(), lr=self.lrD)

		self.G = Generator()
		#self.G = Generator_Union(self.num_cls)
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
#		self.train_hist['GRU_loss'] = []
		self.train_hist['per_epoch_time']=[]
		self.train_hist['total_time']=[]

		if self.gpu_mode:
			self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1).cuda()), Variable(torch.zeros(self.batch_size, 1).cuda())
		else:
			self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1)), Variable(torch.zeros(self.batch_size, 1))
		
		self.fill = torch.zeros([self.num_cls, self.num_cls, self.resl, self.resl])
		for i in range(self.num_cls):
			self.fill[i, i, :, :] =1



		#train
		self.D.train()
		start_time = time.time()
		print('All class is %03d'%self.num_cls)
		for epoch in range(self.epoch):
			self.G.train()
			epoch_start_time = time.time()
			G_acc = 0
			ntr = 0
			for iB, (img_, eeg_, cls_, id_) in enumerate(self.train_loader):
				if iB == self.train_loader.dataset.__len__() // self.batch_size:
					break
				#Latent space
				z_ = torch.rand(self.batch_size, 100)
				
				#make class_label and fake_class_label

				fake_index = torch.randperm(self.batch_size)
				fake_cls = cls_[fake_index]


				y_cls_onehot_ = torch.zeros(self.batch_size, self.num_cls)
				y_cls_onehot_.scatter_(1, cls_.view(-1, 1),1)

				y_fcls_onehot_ = torch.zeros(self.batch_size, self.num_cls)
				y_fcls_onehot_.scatter_(1, fake_cls.view(-1, 1), 1)
				
				
				fake_ans = (cls_==fake_cls).type(torch.FloatTensor)
				
				y_fill_ = self.fill[torch.max(y_cls_onehot_,1)[1].squeeze()]
				y_fake_fill_ = self.fill[torch.max(y_fcls_onehot_, 1)[1].squeeze()]
				

				if self.gpu_mode:
					z_, img_, eeg_, cls_, id_ = Variable(z_.cuda()), Variable(img_.cuda()), Variable(eeg_.cuda()), Variable(cls_.cuda()), Variable(id_.cuda())
					y_cls_onehot_, y_fcls_onehot_, fake_ans = Variable(y_cls_onehot_.cuda()), Variable(y_fcls_onehot_.cuda()), Variable(fake_ans.cuda())
					y_fill_, y_fake_fill_ = Variable(y_fill_.cuda()), Variable(y_fake_fill_.cuda())

				else:
					z_, img_, eeg_, cls_, id_ = Variable(z_), Variable(img_), Variable(eeg_), Variable(cls_), Variable(id_)
					y_cls_onehot_, y_fcls_onehot_, fake_ans = Variable(y_cls_onehot_), Variable(y_fcls_onehot_), Variable(fake_ans)
					y_fill_, y_fake_fill_ = Variable(y_fill_), Variable(y_fake_fill_)


				

				#----Update D_network----#
				self.D_optimizer.zero_grad()

				D_real = self.D(img_, y_fill_)
				D_real_loss = self.BCE_loss(D_real, self.y_real_)
				
				D_real_wrong = self.D(img_, y_fake_fill_)
				D_real_wrong_loss = self.BCE_loss(D_real_wrong, fake_ans)

				G_ = self.G(z_, y_cls_onehot_)
				D_fake = self.D(G_, y_fill_)
				D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)

				#gradient penalty
				if self.use_gp:
					if self.gpu_mode:
						alpha = torch.rand(img_.size()).cuda()
					else:
						alpha = torch.rand(img.size())
					x_hat = Variable(alpha*img_.data + (1-alpha)*G_.data, requires_grad=True)
					pred_hat = self.D(x_hat, y_fill_)
					if self.gpu_mode:
						gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).cuda(),create_graph=True, retain_graph=True, only_inputs=True)[0]
					else:
						gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()), create_grahp=True, retain_graph=True, only_inputs=True)[0]
					
					gradient_penalty = self.lambda_*((gradients.contiguous().view(gradients.size()[0], -1).norm(2,1) -1)**2).mean()
					
					D_loss = D_fake_loss + D_real_loss + D_real_wrong_loss +  gradient_penalty

				else:
					D_loss = D_fake_loss + D_real_loss + D_real_wrong_loss

				self.train_hist['D_loss'].append(D_loss.data[0])

				num_correct_real = torch.sum(D_real > 0.5)
				num_correct_fake = torch.sum(D_fake < 0.5)
				D_acc = float(num_correct_real.data[0]+num_correct_fake.data[0]) / (self.batch_size*2)
			
				D_loss.backward()
				if self.d_trick:
					if D_acc < 0.8:
						self.D_optimizer.step()
				else:
					self.D_optimizer.step()
				

				#---Update G_network---#
				for iG in range(self.n_critic):
					self.G_optimizer.zero_grad()	
					G_ = self.G(z_, y_cls_onehot_)
					D_fake = self.D(G_, y_fill_)
					G_fake_loss = self.BCE_loss(D_fake, self.y_real_)
				
					if self.use_recon:
						G_recon_loss = self.L1_loss(G_, img_)
						G_loss = G_fake_loss + G_recon_loss*100
					else:
						G_loss = G_fake_loss

					if iG == (self.n_critic -1):
						self.train_hist['G_loss'].append(G_loss.data[0])
					G_loss.backward()
					self.G_optimizer.step()

				#---check train result ----#
				if(iB % 100 == 0):
					print('[E%03d]'%(epoch)+'  G_loss :  %.6f '%(G_loss.data[0])+'  D_loss :  %.6f = %.6f + %.6f + %.6f'%(D_loss.data[0], D_fake_loss.data[0], D_real_loss.data[0], D_real_wrong_loss.data[0])+'   D_acc : %.6f'%D_acc)
					self.visualize_results(epoch, z_, y_cls_onehot_, img_, iB)

			#---check train result ----#
			self.train_hist['per_epoch_time'].append(time.time()-epoch_start_time)
			utils.loss_plot(self.train_hist, os.path.join(self.result_dir, self.dataset, self.model_name), self.model_name)
			self.save()
		
		print("Training finish!... save training results")
		self.save()
	
	def test(self):
		#self.load()
		self.GRU.eval()
		print('Test start!')
		
		G_acc = 0
		ntr = 0
		for iB, (eeg_, cls_, id_) in enumerate(self.test_loader):
			if iB == self.test_loader.dataset.__len__() // self.batch_size:
				break
				
			if self.gpu_mode:
				eeg_, cls_, id_ = Variable(eeg_.cuda()), Variable(cls_.cuda()), Variable(id_.cuda())
			else:
				eeg_, cls_, id_ = Variable(eeg_), Variable(cls_), Variable(id_)

			G_real = self.GRU(eeg_)
			
			_, index_G = torch.max(G_real, 1)
			G_acc += float((index_G==cls_).sum())
			ntr += float(cls_.size(0))
			
			print('class label : %02d'%cls_.data[0]+'   GRU_reslut :  %02d'%index_G.data[0])
			
		print('  Total G_acc is %.3f'%float((G_acc/ntr)*100)+'%')


	def visualize_results(self, epoch,z_, y_cls_onehot_, y, iB, fix=True):
		self.G.eval()
		if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
			os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

		tot_num_samples = min(self.sample_num, self.batch_size)
		image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

		if fix:
			""" fixed noise """
			samples = self.G(z_, y_cls_onehot_)
		
		if self.gpu_mode:
			samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
			gt = y.cpu().data.numpy().transpose(0, 2, 3, 1)
		else:
			samples = samples.data.numpy().transpose(0, 2, 3, 1)
			gt = y.data.numpy().transpose(0, 2, 3, 1)

		utils.save_images(samples[:image_frame_dim*image_frame_dim,:,:,:], [image_frame_dim, image_frame_dim], self.result_dir+'/'+self.dataset+'/'+self.model_name+'/'+self.model_name+'_epoch%03d'%epoch+'_I%03d'%iB+'_F.png')
		utils.save_images(gt[:image_frame_dim*image_frame_dim,:,:,:], [image_frame_dim, image_frame_dim], self.result_dir+'/'+self.dataset+'/'+self.model_name+'/'+self.model_name+'_epoch%03d'%epoch+'_I%03d'%iB+'_R.png')


	def save(self):
		save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
		torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))
		#torch.save(self.GRU.state_dict(), os.path.join(save_dir, self.model_name + '_GRU.pkl'))

		with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
			pickle.dump(self.train_hist, f)

	def load(self):
		save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

		self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
		self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))
		self.GRU.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_GRU.pkl')))


