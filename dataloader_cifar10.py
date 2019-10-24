import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import os
import pdb
import sys
import fastNLP
from fastNLP import DataSet , Instance , Vocabulary
import pickle
import random


transform_ori = transforms.Compose([
	transforms.ToTensor() ,
	transforms.Normalize((0.5 , 0.5 , 0.5), (0.5 , 0.5 , 0.5)) ,
])

transform_train_aug1 = transforms.Compose([
	transforms.RandomHorizontalFlip() ,
	transforms.RandomGrayscale() ,
	transforms.ToTensor() ,
	transforms.Normalize((0.5 , 0.5 , 0.5), (0.5 , 0.5 , 0.5)) , 
])

transform_train_aug2 = transforms.Compose([
	transforms.Resize(40) ,
	transforms.RandomHorizontalFlip() ,
	transforms.RandomCrop(32) ,
	transforms.ColorJitter(brightness = 0.5 , contrast = 0.5 , hue = 0.5) ,
	transforms.ToTensor() ,
	transforms.Normalize([0.5 , 0.5 , 0.5], [0.5 , 0.5 , 0.5]) , 
])



def n_crop_test(dataset_location , n_crop = 1):
	os.makedirs(dataset_location , exist_ok = True)

	if n_crop <= 0:
		trs = [transform_ori]
	else:
		trs = [
			transforms.Compose([
				transforms.Resize(34) ,
				transforms.RandomCrop(32) ,
				transforms.ToTensor() ,
				transforms.Normalize((0.5 , 0.5 , 0.5), (0.5 , 0.5 , 0.5)) ,
			])
			for _ in range(n_crop)
		]
	testsets = [
		list(torchvision.datasets.CIFAR10(
			root = dataset_location , 
			train = False, 
			download = True , 
			transform = tr
		) )
		for tr in trs
	]

	#testset[i] : [ [s1, s2 ,...] , lab ]
	testset = [ [ [tes[i][0] for tes in testsets] , testsets[0][i][1] ] for i in range(len(testsets[0])) ]


	return testset

def read_data(dataset_location):
	
	os.makedirs(dataset_location , exist_ok = True)

	trainset_1 	= torchvision.datasets.CIFAR10(root = dataset_location , train = True , download = True , transform = transform_ori)
	trainset_2 	= torchvision.datasets.CIFAR10(root = dataset_location , train = True , download = True , transform = transform_train_aug1)
	trainset_3 	= torchvision.datasets.CIFAR10(root = dataset_location , train = True , download = True , transform = transform_train_aug2)
	testset  	= torchvision.datasets.CIFAR10(root = dataset_location , train = False, download = True , transform = transform_ori)

	train_data = list(trainset_1) + list(trainset_2) + list(trainset_3)
	test_data = list(testset)

	return [train_data , test_data]

def load_data(dataset_location = "datas/cifar-10" , save_path = "processed_datas/" , 
				save_name = "cifar-10" , force_reprocess = False , smallize = False):
	
	if smallize:
		save_name += "-small"

	os.makedirs(save_path , exist_ok = True)
	if save_name:
		save_file = os.path.join(save_path , save_name)
		if os.path.exists(save_file) and not force_reprocess:
			with open(save_file , "rb") as fil:
				dat = pickle.load(fil)
			return dat

	read_datas = read_data(dataset_location)

	dataset_lis = []
	for i in range(len(read_datas)):
		data = read_datas[i]
		dataset = DataSet()
		for s  , lab in data:
			dataset.append ( Instance (s = s , label = lab) )
		dataset.set_input("s")
		dataset.set_target("label")
		if smallize:
			dataset = dataset[:1000]
		dataset_lis.append(dataset)

	dat = {
		"train" : dataset_lis[0],
		"test" 	: dataset_lis[1],
	}

	if save_name:
		with open(save_file , "wb") as fil:
			pickle.dump( dat , fil)
	return dat

if __name__ == "__main__":
	from config import C
	dat = load_data(dataset_location = C.data_load , save_path = C.data_save , save_name = C.data , 
						force_reprocess = C.force_reprocess , smallize = C.smallize)


	pdb.set_trace()