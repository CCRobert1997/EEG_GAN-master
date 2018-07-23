from __future__ import print_function
import os, csv, sys, gzip, torch, time, pickle, argparse
import torch.nn as nn
import numpy as np
import scipy.misc
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pdb
from itertools import islice
import pandas as pd
from numpy import genfromtxt
from collections import defaultdict
import scipy.io as sio

class EEG(Dataset):
	def __init__(self, root_dir, transform = None, _type = None):
		self.filenames = []
		self.root_dir = '../../eeg/eeg_128/eeg_matlab/'#root_dir
		self.transform = transform
		self.type = _type

		print('Loading EEG matLab metadata...')
		sys.stdout.flush()
		time_start = time.time()

		fname_cache = 'EEG_mat_cache.txt'
		fname_cache_train, fname_cache_test = './cache/train.txt' , './cache/test.txt'
		
		if os.path.exists(fname_cache):
			self.filenames = open(fname_cache).read().splitlines()
			print('Already cache file exists! Load from here...')
		else:
			self.filenames = [os.path.join(dirpath,f) for dirpath, dirnames, files in os.walk(self.root_dir) for f in files ]
			with open(fname_cache, 'w') as f:
				for fname in self.filenames:
					f.write(fname+'\n')
			print('---Make cache file---')
		

		self.filenames_train = []
		self.filenames_test = []
		self.EEG_dict = defaultdict(list)
		if os.path.exists(fname_cache_train) and os.path.exists(fname_cache_test):
			self.filenames_train = open(fname_cache_train).read().splitlines()
			self.filenames_test = open(fname_cache_test).read().splitlines()
			print('Already cache file exists! Load from here...')
		else:
			for i, item in enumerate(self.filenames):
				cls = os.path.basename(item).split('_')[0]
				self.EEG_dict[cls].append((item))

			for iB, key in enumerate(self.EEG_dict):
				length = len(self.EEG_dict[key])
				train_len = int(length*0.8)
				for i in range(length):
					basename = self.EEG_dict[key][i]
					if i<= train_len:
						self.filenames_train.append(basename)
					else:
						self.filenames_test.append(basename)
		'''
		if self.type == 'train':
			self.cls = sorted(set([os.path.basename(f).split('_')[0] for f in self.filenames_train]))
			self.id = sorted(set([os.path.dirname(f).split('/')[-1] for f in self.filenames_train]))
		else:
			self.cls = sorted(set([os.path.basename(f).split('_')[0] for f in self.filenames_test]))
			self.id = sorted(set([os.path.dirname(f).split('/')[-1] for f in self.filenames_test]))
		'''

		
		with open(fname_cache_train, 'w')as f:
			for fname in self.filenames_train:
				f.write(fname+'\n')
		with open(fname_cache_test, 'w')as f:
			for fname in self.filenames_test:
				f.write(fname+'\n')

		self.cls = sorted(set([os.path.basename(f).split('_')[0] for f in self.filenames]))
		self.id = sorted(set([os.path.dirname(f).split('/')[-1] for f in self.filenames]))

		self.cls_map = {}
		self.id_map = {}
		for i, cls in enumerate(self.cls):
			self.cls_map[cls] = i
		for i, id_ in enumerate(self.id):
			self.id_map[id_] = i
		
		

	def __len__(self):
		if self.type == 'train':
			return len(self.filenames_train)
		elif self.type == 'test':
			return len(self.filenames_test)
	
	def __getitem__(self, idx):
		n_ch = 128
		len_ = 512
		eeg_ = np.zeros((n_ch, len_))
		filenames = []
		if self.type == 'train':
			filenames = self.filenames_train
		else:
			filenames = self.filenames_test
		
		cls_ = os.path.basename(filenames[idx]).split('_')[0]
		id_ = os.path.dirname(filenames[idx]).split('/')[-1]
			
		curr_x = sio.loadmat(filenames[idx])['x']
		curr_x = np.swapaxes(curr_x, 0, 1)
		eeg_[:,:min(curr_x.shape[1], 440)] = curr_x[:,40:480]
		
		#normalize
		#eeg_ = eeg_ / np.linalg.norm(eeg_)

		eeg_ = torch.FloatTensor(eeg_)
		cls_ = self.cls_map[cls_]
		id_ = self.id_map[id_]

		return eeg_, cls_, id_

class EEG_pytorch(Dataset):
	def __init__(self, root_dir, transform = None, _type = None, num_cls = None):
		self.filenames = []
		self.transform=transform
		self.root_dir = '../../eeg/eeg_128/data/'
		self.Img_root_dir = '../../ImageNet/Data/'
		self.data_path = 'eeg_signals_128_sequential_band_all_with_mean_std.pth'
		self.split_path = 'splits_by_image.pth'
		self.data = torch.load(self.root_dir+self.data_path)
		self.split = torch.load(self.root_dir+self.split_path)

		self.mean = self.data['means']
		self.stdev = self.data['stddevs']

		self.labels = self.split['splits'][0][_type]
		self.images = self.data['images']
		self.eeg = []

		for label in self.labels:
			#image check
			data = self.data['dataset'][label]
			img_idx = data['image']
			img_path = str(self.images[img_idx])
			dir_path = str(self.images[img_idx].split('_')[0])
			path = os.path.join(self.Img_root_dir, dir_path, img_path+'.JPEG')
			if os.path.exists(path):
				self.eeg.append(self.data['dataset'][label])
		
		#assert len(self.eeg)==len(self.labels)
		self.dlen = 440

	def __len__(self):
		return len(self.eeg)
	
	def __getitem__(self, idx):
		nch = 128
		dlen = self.dlen
		eeg_ = np.zeros((nch, dlen))
		cls_ = self.eeg[idx]['label']
		sub_ = self.eeg[idx]['subject']
		img_idx = self.eeg[idx]['image']
		img_path = str(self.images[img_idx])
		dir_path = str(self.images[img_idx].split('_')[0]) 

		path = os.path.join(self.Img_root_dir, dir_path, img_path+'.JPEG')

		image = Image.open(path)
		image = image.convert('RGB')

		if self.transform:
			image = self.transform(image)

		eeg_ = torch.FloatTensor(eeg_)
		eeg_[:, :min(int(self.eeg[idx]['eeg'].shape[1]), dlen)] = self.eeg[idx]['eeg'][:, 40:40+dlen]

		#image_path = 
		

		return image, eeg_, cls_, sub_
		#return eeg_, cls_, sub_


class ImageNet(Dataset):
	def __init__(self, root_dir, transform = None, _type = None, num_cls = None):
		self.filenames = []
		self.root_dir = root_dir#../../../ImageNet/ILSVRC/Data/Det
		self.transform = transform
		self.type = _type
		self.num_cls = num_cls+1

		print('Loading ImageNet metadata...')
		sys.stdout.flush()
		time_start = time.time()

		#make cache text file
		fname_cache = 'ImageNet_cache.txt'
		if os.path.exists(fname_cache):
			self.filenames = open(fname_cache).read().splitlines()
			print('Already cache file exists! Load from here...')
		else:
			if self.type == 'train':
				path = os.path.join(root_dir, self.type,'ILSVRC2013_train')
			else:
				path = os.path.join(root_dir, self.type)
			
			self.filenames = [ os.path.join(dirpath,f) for _,( dirpath, dirnames,files) in zip(range(self.num_cls), os.walk(path)) for f in files if f.endswith('.JPEG')]
			print('---Making cache_file.txt----')

			with open(fname_cache, 'w') as f:
				for fname in self.filenames:
					f.write(fname+'\n')
			print('Done! cached in {}'.format(fname_cache))



		#get ImageNet file path
		self.cls = sorted(set( [os.path.basename(f).split('_')[0] for f in self.filenames]))
		self.cls_map = {}
		for i, cls in enumerate(self.cls):
			self.cls_map[cls] = i	
		print('Loading ImageNet done!')

		
	
	def __len__(self):
		return len(self.filenames)

	def __getitem__(self, idx):
		#load image
		filename = self.filenames[idx]
		image = Image.open(filename)
		image = image.convert('RGB')
		cls_ = os.path.basename(filename).split('_')[0]
		if self.transform:
			image = self.transform(image)

		cls = self.cls_map[cls_]
		
		return image, cls



class EEG_ImageNet(Dataset):
	def __init__(self, root_dir, transform = None, _type = None, mini = 0):
		self.EEG_filenames=[]
		self.root_dir = root_dir #../../../EEG~
		self.transform = transform
		self.type = _type #train or test
		self.Img_root_dir = '../../ImageNet/ILSVRC/Data/DET/train/ILSVRC2013_train'
		self.Spc_root_dir = '../../eegImagenet/mindbigdata-imagenet-in-v1.0/MindBigData-Imagenet-v1.0-Imgs'
		self.mini = mini
		limit = -1	
		self.EEG_dic=defaultdict(list)
		
		self.EEG_filenames_train = []
		self.EEG_filenames_test = []
		self.Img_filenames_train = []
		self.Img_filenames_test = []
		self.Spc_filenames_train = []
		self.Spc_filenames_test = []


		print('Loading EEG-ImageNet metadata...')

		fname_cache = 'EEG_cache.txt'
		if os.path.exists(fname_cache):
			self.EEG_filenames = open(fname_cache).read().splitlines()
			print('Already EEG_cache file exists! Load from here...')
		else:
			if self.type == 'train':
				path = root_dir
			else:
				path = root_dir

			self.EEG_filenames = [os.path.join(dirpath, f) for (dirpath, dirnames, files) in os.walk(path) for f in files if f.endswith('csv')]
			print('----MAking cache_file.txt----')
			
			with open(fname_cache, 'w') as f:
				for fname in self.EEG_filenames:
					f.write(fname+'\n')
			print('Done! cached in {}'.format(fname_cache))
				
		fname_cache_eeg_train, fname_cache_eeg_test = './cache/EEG_cache_train.txt', './cache/EEG_cache_test.txt'
		fname_cache_img_train, fname_cache_img_test = './cache/Img_cache_train.txt', './cache/Img_cache_test.txt'
		fname_cache_spc_train, fname_cache_spc_test = './cache/Spc_cache_train.txt', './cache/Spc_cache_test.txt'

		if self.mini !=  0:
			limit = self.mini
			fname_cache_eeg_train = os.path.join(os.path.dirname(fname_cache_eeg_train), str(limit)+'_'+os.path.basename(fname_cache_eeg_train))
			fname_cache_eeg_test = os.path.join(os.path.dirname(fname_cache_eeg_test), str(limit)+'_'+os.path.basename(fname_cache_eeg_test))
			fname_cache_img_train = os.path.join(os.path.dirname(fname_cache_img_train), str(limit)+'_'+os.path.basename(fname_cache_img_train))
			fname_cache_img_test = os.path.join(os.path.dirname(fname_cache_img_test), str(limit)+'_'+os.path.basename(fname_cache_img_test))
			fname_cache_spc_train = os.path.join(os.path.dirname(fname_cache_spc_train), str(limit)+'_'+os.path.basename(fname_cache_spc_train))
			fname_cache_spc_test = os.path.join(os.path.dirname(fname_cache_spc_test), str(limit)+'_'+os.path.basename(fname_cache_spc_test))


		if os.path.exists(fname_cache_eeg_train):
			self.EEG_filenames_train = open(fname_cache_eeg_train).read().splitlines()
			self.EEG_filenames_test = open(fname_cache_eeg_test).read().splitlines()
			self.Img_filenames_train = open(fname_cache_img_train).read().splitlines()
			self.Img_filenames_test = open(fname_cache_img_test).read().splitlines()
			self.Spc_filenames_train = open(fname_cache_spc_train).read().splitlines()
			self.Spc_filenames_test = open(fname_cache_spc_test).read().splitlines()
			print('Already cache file exists! Load from here...')
		else:		
			for i, item in enumerate(self.EEG_filenames):
				cls = os.path.basename(item).split('_')[3]
				target = os.path.basename(item).split('_')[4]
				self.EEG_dic[cls].append((item, target))
		
			for iB, key in enumerate(self.EEG_dic):
				length = len(self.EEG_dic[key])
				train_len = int(length*0.8)

				#control num_cls
				if iB == limit:
					break

				for i in range(length):
					basename = os.path.basename(self.EEG_dic[key][i][0]).split('.')[0]
					e_path = self.EEG_dic[key][i][0]
					s_path = os.path.join(self.Spc_root_dir, basename+'.png')
					i_path = os.path.join(self.Img_root_dir, key, key+'_'+self.EEG_dic[key][i][1]+'.JPEG')
					#pdb.set_trace()
					if i <= train_len:
						self.EEG_filenames_train.append(e_path)
						self.Img_filenames_train.append(i_path)
						self.Spc_filenames_train.append(s_path)
					else:
						self.EEG_filenames_test.append(e_path)
						self.Img_filenames_test.append(i_path)
						self.Spc_filenames_test.append(s_path)

			with open(fname_cache_eeg_train, 'w') as f:
				for fname in self.EEG_filenames_train:
					f.write(fname+'\n')
			with open(fname_cache_eeg_test, 'w') as f:
				for fname in self.EEG_filenames_test:
					f.write(fname+'\n')
			with open(fname_cache_img_train, 'w') as f:
				for fname in self.Img_filenames_train:
					f.write(fname+'\n')
			with open(fname_cache_img_test, 'w') as f:
				for fname in self.Img_filenames_test:
					f.write(fname+'\n')
			with open(fname_cache_spc_train, 'w') as f:
				for fname in self.Spc_filenames_train:
					f.write(fname+'\n')
			with open(fname_cache_spc_test, 'w') as f:
				for fname in self.Spc_filenames_test:
					f.write(fname+'\n')
	
		'''
		if self.mini:
			#train
			self.EEG_filenames_train = self.EEG_filenames_train[:100]
			self.Img_filenames_train = self.Img_filenames_train[:100]
			self.Spc_filenames_train = self.Spc_filenames_train[:100]
			
			self.EEG_filenames_test = self.EEG_filenames_test[:20]
			self.Img_filenames_test = self.Img_filenames_test[:20]
			self.Spc_filenames_test = self.Spc_filenames_test[:20]
		'''
	
		#get EEG csv file path
		if self.type == 'train': 
			self.cls = sorted(set([os.path.basename(f).split('_')[3] for f in self.EEG_filenames_train]))
		else:
			self.cls = sorted(set([os.path.basename(f).split('_')[3] for f in self.EEG_filenames_test]))
		self.cls_map = {}
		for i, cls in enumerate(self.cls):
			self.cls_map[cls] = i
		print('Loading Meta EEG csv done!')
	
	def __len__(self):
		if self.type == 'train':
			return len(self.EEG_filenames_train)
		else:
			return len(self.EEG_filenames_test)

	def __getitem__(self, idx):
		if self.type =='train':
			#load csv data
			e_path = self.EEG_filenames_train[idx]
			cls_ = os.path.basename(e_path).split('_')[3]
			csv_data = genfromtxt(e_path, delimiter=',')
			#if you want normalize please normalize csv_data here
			eeg_data = torch.FloatTensor(csv_data[:, 1:361]) #I should fill the zero last array which shorter than threshold
			
			#Image has mush bigger size so we should differ transform between Img and Spc

			i_path = self.Img_filenames_train[idx]
			image = Image.open(i_path).convert('RGB')

			s_path = self.Spc_filenames_train[idx]
			spc = Image.open(s_path).convert('RGB')
			
			if self.transform:
				image = self.transform(image)
				spc = self.transform(spc)

			cls = self.cls_map[cls_]
			#pdb.set_trace()
			#print(eeg_data.shape, image.shape, spc.shape, cls, sep='  ')
			return eeg_data, image, spc, cls

		elif self.type == 'test':
			e_path = self.EEG_filenames_test[idx]
			cls_ = os.path.basename(e_path).split('_')[3]
			csv_data = genfromtxt(e_path, delimiter=',')

			eeg_data = torch.FloatTensor(csv_data[:, 1:351])

			i_path = self.Img_filenames_test[idx]
			image = Image.open(i_path).convert('RGB')

			s_path = self.Spc_filenames_test[idx]
			spc = Image.open(s_path).convert('RGB')

			if self.transform:
				image = self.transform(image)
				spc = self.transform(spc)

			cls = self.cls_map[cls_]
			return eeg_data, image, spc, cls

		





def print_network(net):
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	print(net)
	print('Total number of parameters: %d' % num_params)

def save_images(images, size, image_path):
	return imsave(images, size, image_path)

def imsave(images, size, path):
	image = np.squeeze(merge(images, size))
	return scipy.misc.imsave(path, image)

def merge(images, size):
	h, w = images.shape[1], images.shape[2]
	if (images.shape[3] in (3,4)):
		c = images.shape[3]
		img = np.zeros((h * size[0], w * size[1], c))
		for idx, image in enumerate(images):
			i = idx % size[1]
			j = idx // size[1]
			img[j * h:j * h + h, i * w:i * w + w, :] = image
		return img
	elif images.shape[3]==1:
		img = np.zeros((h * size[0], w * size[1]))
		for idx, image in enumerate(images):
			i = idx % size[1]
			j = idx // size[1]
			img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
		return img
	else:
		raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def generate_animation(path, num):
	images = []
	for e in range(num):
		img_name = path + '_epoch%03d' % (e+1) + '.png'
		images.append(imageio.imread(img_name))
	imageio.mimsave(path + '_generate_animation.gif', images, fps=5)

def loss_plot(hist, path='.', model_name='model', y_max=None, use_subplot=False, keys_to_show=[] ):
	try:
		x = range(len(hist['D_loss']))
	except:
		keys = hist.keys()
		lens = [ len(hist[k]) for k in keys if 'loss' in k ]
		maxlen = max(lens)
		x = range(maxlen)

	if use_subplot:
		f, axarr = plt.subplots(2, sharex=True)
		
	plt.xlabel('Iter')
	plt.ylabel('Loss')
	plt.tight_layout()

	if len(keys_to_show) == 0:
		keys_to_show = hist.keys()
	for key,value in hist.iteritems():
		if 'time' in key or key not in keys_to_show:
			continue
		y = value
		if len(x) != len(y):
			print('[warning] loss_plot() found mismatching dimensions: {}'.format(key))
			continue
		if use_subplot and 'acc' in key:
			axarr[1].plot(x, y, label=key)
		elif use_subplot:
			axarr[0].plot(x, y, label=key)
		else:
			plt.plot(x, y, label=key)

	if use_subplot:
		axarr[0].legend(loc=1)
		axarr[0].grid(True)
		axarr[1].legend(loc=1)
		axarr[1].grid(True)
	else:
		plt.legend(loc=1)
		plt.grid(True)


	if y_max is not None:
		if use_subplot:
			x_min, x_max, y_min, _ = axarr[0].axis()
			axarr[0].axis( (x_min, x_max, -y_max/20, y_max) )
		else:
			x_min, x_max, y_min, _ = plt.axis()
			plt.axis( (x_min, x_max, -y_max/20, y_max) )

	path = os.path.join(path, model_name + '_loss.png')

	plt.savefig(path)

	plt.close()

def initialize_weights(net):
	for m in net.modules():
		if isinstance(m, nn.Conv2d):
			m.weight.data.normal_(0, 0.02)
			if m.bias is not None:
				m.bias.data.zero_()
		elif isinstance(m, nn.ConvTranspose2d):
			m.weight.data.normal_(0, 0.02)
			if m.bias is not None:
				m.bias.data.zero_()
		elif isinstance(m, nn.Conv3d):
			nn.init.xavier_uniform(m.weight)
		elif isinstance(m, nn.ConvTranspose3d):
			nn.init.xavier_uniform(m.weight)
		#elif isinstance(m, nn.GRU):
		#	nn.init.xavier_uniform(m.weight)
		elif isinstance(m, nn.Linear):
			m.weight.data.normal_(0, 0.02)
			m.bias.data.zero_()


class Flatten(nn.Module):
	def __init__(self):
		super(Flatten, self).__init__()

	def forward(self, x):
		return x.view(x.size(0), -1)


class Inflate(nn.Module):
	def __init__(self, nDims2add):
		super(Inflate, self).__init__()
		self.nDims2add = nDims2add

	def forward(self, x):
		shape = x.size() + (1,)*self.nDims2add
		return x.view(shape)


def parse_args():
	desc = "plot loss"
	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('--fname_hist', type=str, default='', help='history path', required=True)
	parser.add_argument('--fname_dest', type=str, default='.', help='filename of png')
	return parser.parse_args()

if __name__ == '__main__':
	opts = parse_args()
	with open( opts.fname_hist ) as fhandle:
		history = pickle.load(fhandle)
		loss_plot( history, opts.fname_dest )
