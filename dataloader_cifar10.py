import torch as tc
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

def n_crop_test(dataset_location , n_crop = 1):
	os.makedirs(dataset_location , exist_ok = True)

	normalize = transforms.Normalize(mean = [0.491 , 0.482 , 0.447] , std = [0.247 , 0.243 , 0.262])

	if n_crop <= 0:
		trs = [
			transforms.Compose([
				transforms.ToTensor(),
				normalize,
			]),
		]
	else:
		trs = [
			transforms.Compose([
				transforms.RandomCrop(32, padding = 4),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				normalize,
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

def load_data(dataset_location = "./datas"):

	os.makedirs(dataset_location , exist_ok = True)

	transform_train = transforms.Compose([
		transforms.RandomCrop(32 , padding = 4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(mean = [0.491 , 0.482 , 0.447] , std = [0.247 , 0.243 , 0.262]),
	])

	transform_test  = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean = [0.491 , 0.482 , 0.447] , std = [0.247 , 0.243 , 0.262]),
	])

	train_data = torchvision.datasets.CIFAR10(root = dataset_location , train = True  , 
			download = True , transform = transform_train)

	test_data  = torchvision.datasets.CIFAR10(root = dataset_location , train = False , 
			download = True , transform = transform_test)

	return {
		"train" : train_data ,
		"test"  : test_data  ,
	}


if __name__ == "__main__":
	from config import C
	dat = load_data(dataset_location = C.data_path)


	pdb.set_trace()