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
	def __init__(self, num_cls):
		super(Encoder, self).__init__()
		self.input_dim = 3
		self.input_height = 64
		self.input_width = 64
		self.output_dim = num_cls#50

		self.conv = nn.Sequential(
			nn.Conv2d(self.input_dim, 32, 3, 1, 1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32, 64, 3, 1, 1),
			nn.BatchNorm2d(64),
			nn.ReLU(),

			nn.Conv2d(64, 64, 4, 2, 1), # 64 -> 31
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 64, 3, 1, 1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 128, 3, 1, 1),
			nn.BatchNorm2d(128),
			nn.ReLU(),

			nn.Conv2d(128, 128, 4, 2, 1),# 31->15
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Conv2d(128, 128, 3, 1, 1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Conv2d(128, 256, 3, 1, 1),
			nn.BatchNorm2d(256),
			nn.ReLU(),

			nn.Conv2d(256, 256, 4, 2, 1),# 8->4
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.Conv2d(256, 160, 3, 1, 1),
			nn.BatchNorm2d(160),
			nn.ReLU(),
			nn.Conv2d(160, 320, 3, 1, 1),
			nn.BatchNorm2d(320),
			nn.ReLU(),

			nn.AvgPool2d(6)
			#nn.Sigmoid(),
		)

		self.fc = nn.Sequential(
			nn.Linear(320, self.output_dim),
			#nn.Sigmoid()
		)

		utils.initialize_weights(self)

	def forward(self, input):
		x = self.conv(input).squeeze(3).squeeze(2)
		x = self.fc(x)
		return x

class GRU_Encoder(nn.Module):
	def __init__(self, num_cls):
		super(GRU_Encoder, self).__init__()
		self.hidden_dim = 64
		self.input_dim = 128
		self.output_dim = 128#num_cls#50 # class_num or feature dimension(for concat)
	
		self.GRU = nn.GRU(self.input_dim, self.hidden_dim, num_layers = 2,  batch_first = True, dropout=0.5)
		self.fc = nn.Sequential(
			nn.Linear(self.hidden_dim , self.output_dim),
			nn.ReLU(),
			#nn.Sigmoid()
		)
		self.fc2 = nn.Sequential(
			nn.Linear(self.output_dim, 40),
		)
		#utils.initialize_weights(self)

	def forward(self, feature):

		feature = feature.transpose(1,2)#dimension change : batch x time x dimension	
		x, hidden = self.GRU(feature)
		x = x.select(1, x.size(1)-1).contiguous()
		x = x.view(-1, self.hidden_dim)
		result = self.fc(x)
		result = self.fc2(result)
		return result


class LSTM(nn.Module):
	#I should divide one channel each and pass through LSTM and lst concat each channel(dim)
	def __init__(self, batch_size):
		super(LSTM, self).__init__()
		self.hidden_dim = 50
		self.embedding_dim = 32
		self.input_dim = 5
		self.output_dim = 10
		self.batch_size = batch_size

		self.embedding = nn.Embedding(self.input_dim, self.embedding_dim)
		self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim)
		self.hidden2label = nn.Linear(self.hidden_dim, self.output_dim)

		self.hidden = self.init_hidden()
	
	def init_hidden(self):
		h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
		c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
		return (h0, c0)

	def forward(self, sentence):
		pdb.set_trace()
		embeds = self.embedding(sentence)
		x = embeds.view(len(sentence), self.batch_size, -1)
		lstm_out, self.hidden = self.lstm(x, self.hidden)
		y  = self.hidden2label(lstm_out[-1])
		return y


class RNN(object):
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
		self.mini = args.num_cls
		self.enc_dim = 100
#		self.num_cls = 100

		#load dataset
		'''
		self.data_loader = DataLoader(utils.EEG(root_dir = self.dataroot_EEG_dir, transform=transforms.Compose([transforms.Scale(64),transforms.RandomCrop(64),transforms.ToTensor()]),_type = self.type), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

		self.test_loader = DataLoader(utils.EEG(root_dir = self.dataroot_EEG_dir, transform=None, _type = 'test'), batch_size = self.batch_size, shuffle=True, num_workers=self.num_workers)
		'''
		self.train_loader = DataLoader(utils.EEG_pytorch(root_dir = './', transform = None, _type = 'train'), batch_size = self.batch_size , shuffle=True, num_workers = self.num_workers)

		self.test_loader = DataLoader(utils.EEG_pytorch(root_dir = './', transform = None, _type = 'test'), batch_size = self.batch_size, shuffle=True, num_workers = self.num_workers)


		#self.num_cls = self.data_loader.dataset.num_cls
		self.num_cls = 40#len(self.data_loader.dataset.cls_map)

		self.GRU = GRU_Encoder(self.num_cls)

		self.GRU_optimizer = optim.Adam(self.GRU.parameters(), lr=self.lrD)

		if self.gpu_mode:
			self.GRU = self.GRU.cuda()
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
		self.train_hist['per_epoch_time']=[]
		self.train_hist['total_time']=[]

		if self.gpu_mode:
			self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1).cuda()), Variable(torch.zeros(self.batch_size, 1).cuda())
		else:
			self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1)), Variable(torch.zeros(self.batch_size, 1))

		#train
		self.GRU.train()
		start_time = time.time()
		print('All class is %03d'%self.num_cls)
		for epoch in range(self.epoch):
			self.GRU.train()
			epoch_start_time = time.time()
			G_acc = 0
			ntr = 0
			for iB, (eeg_, cls_, id_) in enumerate(self.train_loader):
				if iB == self.train_loader.dataset.__len__() // self.batch_size:
					break
				
				if self.gpu_mode:
					eeg_, cls_, id_ = Variable(eeg_.cuda()), Variable(cls_.cuda()), Variable(id_.cuda())
				else:
					eeg_, cls_, id_ = Variable(eeg_), Variable(cls_), Variable(id_)


				#----Update GRU_network----#
				self.GRU_optimizer.zero_grad()
				G_real = self.GRU(eeg_)
				G_loss = self.CE_loss(G_real, cls_)
				self.train_hist['G_loss'].append(G_loss.data[0])
				G_loss.backward()
				self.GRU_optimizer.step()

				_, index_G = torch.max(G_real, 1)
				G_acc += float((index_G==cls_).sum())
				ntr += float(cls_.size(0))
				#---check train result ----#
				if(iB % 100 == 0):
					print('[E%03d]'%(epoch)+'  GRU_loss : %.6f'%G_loss.data[0]+'   GRU_acc :  %.6f'%(float(G_acc/ntr)*100)+'%')

			#---check train result ----#
			self.train_hist['per_epoch_time'].append(time.time()-epoch_start_time)
			utils.loss_plot(self.train_hist, os.path.join(self.result_dir, self.dataset, self.model_name), self.model_name)
			self.test()
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
			
			#print('class label : %02d'%cls_.data[0]+'   GRU_reslut :  %02d'%index_G.data[0])
			
		print('  Total G_acc is %.3f'%float((G_acc/ntr)*100)+'%')


	def visualize_results(self, epoch, eeg, spc, z, y, iB, fix=True):
		self.G.eval()
		if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
			os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

		tot_num_samples = min(self.sample_num, self.batch_size)
		image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

		if fix:
			""" fixed noise """
			samples = self.G(eeg, spc, z)
		
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

		torch.save(self.GRU.state_dict(), os.path.join(save_dir, self.model_name + '_GRU.pkl'))

		with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
			pickle.dump(self.train_hist, f)

	def load(self):
		save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

#		self.E.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_E.pkl')))
		self.GRU.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_GRU.pkl')))


