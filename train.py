import os , sys

from config import C , logger

import fastNLP
from fastNLP import Vocabulary , DataSet , Instance , Tester
from fastNLP import Trainer , AccuracyMetric , CrossEntropyLoss
import torch.nn as nn
import torch as tc
from torch.optim import Adadelta
import numpy as np
import random
import pickle
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms

from model.multi_dim_transformer import Model as MD_Transformer
from model.resnet import Model as ResNet
from model.third_party import ResNet_56_3p
from dataloader_cifar10 import load_data as load_data_cifar_10
from optim import MyAdam , MySGD
from utils.confirm_tensor import tensor_feature

import pdb



#---------------------------------------------------------------------------------------------------
#Get data
data_loaders = {
	"cifar-10" 			: load_data_cifar_10,
}

data = data_loaders[C.data](dataset_location = C.data_path)

train_data , test_data = data["train"] , data["test"]

trainloader = tc.utils.data.DataLoader(train_data , batch_size = C.batch_size , shuffle = True , num_workers = 2)
testloader  = tc.utils.data.DataLoader(test_data  , batch_size = 100 		  , shuffle = False, num_workers = 2)
#batch_size should be able to be deviced by 10000

logger.log ("Data load done.")

#---------------------------------------------------------------------------------------------------
#Get model

models = {
	"transformer" 	: MD_Transformer,
	"resnet" 		: ResNet,
	"3p-resnet" 	: ResNet_56_3p,
}
model = models[C.model]
net = model(num_class = 10 , input_size = [32,32] ,
	**{x : C.__dict__[x] for x in model.choose_kwargs()}
)

logger.log ("Creat network done.")

#---------------------------------------------------------------------------------------------------
#Valid

def valid(net , valid_data , epoch_num = 0):

	net = net.eval()

	tota_hit = 0
	good_hit = 0

	pbar = tqdm(testloader , ncols = 70)	
	pbar.set_description_str("(Epoch %d) Testing " % (epoch_num+1))

	for (xs, goldens) in pbar:

		xs = xs.cuda()
		goldens = goldens.cuda()
		ys = net(xs)["pred"]

		got = tc.max(ys , -1)[1]
		good_hit += int((goldens == got).long().sum())
		tota_hit += len(goldens)
		pbar.set_postfix_str("Valid Acc : %d/%d = %.4f%%" % (good_hit , tota_hit , 100 * good_hit / tota_hit))

	return good_hit , tota_hit

#---------------------------------------------------------------------------------------------------
#Optimizer control



#---------------------------------------------------------------------------------------------------
#Train
logger.log("Training start.")
logger.log("--------------------------------------------------------------------")

#variables about model saving
model_save_path = os.path.join(C.model_path , C.model_save)
best_acc = 0.
best_epoch = 0.

#optimizer
optims = {
	"myadam" : lambda : MyAdam(params = net.parameters() , d_model = C.d_model , 
					n_warmup_steps = 4000 , init_steps = C.init_steps , step_size = C.step_size) ,
	"mysgd"  : lambda : MySGD (params = net.parameters() , lr = C.lr) , 
	"adam" 	 : lambda : tc.optim.Adam(params = net.parameters() , lr = C.lr) , 
	"sgd" 	 : lambda : tc.optim.SGD (params = net.parameters() , lr = C.lr) , 
}

optim = optims[C.optim]()


#loss function
loss_func = nn.CrossEntropyLoss()

net = net.cuda()

tot_step = 0
for epoch_num in range(C.n_epochs):

	net = net.train()
	tota_hit = 0
	good_hit = 0
	pbar = tqdm(trainloader , ncols = 70)
	pbar.set_description_str("(Epoch %d) Training" % (epoch_num+1))
	for (xs, goldens) in pbar:

		xs = xs.cuda()
		goldens = goldens.cuda()
		ys = net(xs)["pred"]

		loss = loss_func(ys , goldens)
		optim.zero_grad()
		loss.backward()
		optim.step()

		got = tc.max(ys , -1)[1]
		good_hit += int((goldens == got).long().sum())
		tota_hit += len(goldens)
		pbar.set_postfix_str("Train Acc : %d/%d = %.4f%%" % (good_hit , tota_hit , 100 * good_hit / tota_hit))
		tot_step += 1

	valid_res = valid(net , data[C.valid_data] , epoch_num = epoch_num)

	logger.log("Epoch %d ended." % (epoch_num + 1))
	logger.log("Train Acc : %d/%d = %.4f%%" % (good_hit , tota_hit , 100 * good_hit / tota_hit))
	logger.log("Valid Acc : %d/%d = %.4f%%" % (valid_res[0] , valid_res[1] , 100 * valid_res[0] / valid_res[1]))
	logger.log("now total step = %d" % (tot_step))

	valid_acc = valid_res[0] / valid_res[1]
	if valid_acc > best_acc:
		best_acc = valid_acc
		best_epoch = epoch_num

		net = net.cpu()
		with open(model_save_path , "wb") as fil:
			pickle.dump(net , fil)
		net = net.cuda()
		logger.log("Got new best acc. Model saved.")

	logger.log("--------------------------------------------------------------------")

logger.log("Best Accurancy: %.4f%% in epoch %d" % (best_acc , best_epoch))