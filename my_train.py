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

from model.multi_dim_transformer import Model as MD_Transformer
from model.resnet import Model as ResNet
from model.third_party import ResNet_56_3p
from dataloader_cifar10 import load_data as load_data_cifar_10
from optim import MyAdam , MySGD

import pdb



#---------------------------------------------------------------------------------------------------
#Get data
data_loaders = {
	"cifar-10" 			: load_data_cifar_10,
}

data = data_loaders[C.data](
	dataset_location = C.data_load , save_path = C.data_save , save_name = C.data , 
	force_reprocess = C.force_reprocess , smallize = C.smallize , 
)

train_data , test_data = data["train"] , data["test"]

if C.valid_size > 0:
	C.valid_size = min(C.valid_size , len(data["train"]) // 10)
	data["valid"] = data["train"][:C.valid_size]
	data["train"] = data["train"][C.valid_size:]

logger.log ("Data load done.")
logger.log ("train size = %d , vali size = %d test size = %d" % (len(data["train"]) , len(data["valid"]) , len(data["test"])))
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
#Train & Test

def valid(net , valid_data , epoch_num = 0):
	net = net.eval()
	n_batchs = len(valid_data) // bs
	tota_hit = 0
	good_hit = 0
	pbar = tqdm(range(n_batchs) , ncols = 70)	
	pbar.set_description_str("(Epoch %d) Testing " % (epoch_num+1))

	for batch_num in pbar:
		batch_data = valid_data[batch_num * bs : (batch_num+1) * bs]

		xs = tc.cat( [x["s"].unsqueeze(0) for x in batch_data] , dim = 0).cuda(C.gpus[0])
		goldens = tc.LongTensor([ x["label"] for x in batch_data ]).cuda(C.gpus[0])

		ys = net(xs)["pred"]

		got = tc.max(ys , -1)[1]
		#pdb.set_trace()
		good_hit += int((goldens == got).long().sum())
		tota_hit += len(goldens)
		pbar.set_postfix_str("Valid Acc : %d/%d = %.4f%%" % (good_hit , tota_hit , 100 * good_hit / tota_hit))

	return good_hit , tota_hit



optims = {
	"myadam" : lambda : MyAdam(params = net.parameters() , d_model = C.d_model , 
					n_warmup_steps = 4000 , init_steps = C.init_steps , step_size = C.step_size) ,
	"mysgd"  : lambda : MySGD (params = net.parameters() , lr = C.lr) , 
	"adam" 	 : lambda : tc.optim.Adam(params = net.parameters() , lr = C.lr) , 
	"sgd" 	 : lambda : tc.optim.SGD (params = net.parameters() , lr = C.lr) , 
}

optim = optims[C.optim]()
loss_func = nn.CrossEntropyLoss()

net = net.cuda(C.gpus[0])

n_epochs = C.n_epochs
bs = C.batch_size
n_batchs = len(train_data) // bs
for epoch_num in range(n_epochs):
	net = net.train()
	tota_hit = 0
	good_hit = 0
	pbar = tqdm(range(n_batchs) , ncols = 70)
	pbar.set_description_str("(Epoch %d) Training" % (epoch_num+1))
	for batch_num in pbar:
		batch_data = train_data[batch_num * bs : (batch_num+1) * bs]

		xs = tc.cat( [x["s"].unsqueeze(0) for x in batch_data] , dim = 0).cuda(C.gpus[0])
		goldens = tc.LongTensor([ x["label"] for x in batch_data ]).cuda(C.gpus[0])
		ys = net(xs)["pred"]

		#pdb.set_trace()


		loss = loss_func(ys , goldens)
		optim.zero_grad()
		loss.backward()
		optim.step()

		got = tc.max(ys , -1)[1]
		#pdb.set_trace()
		good_hit += int((goldens == got).long().sum())
		tota_hit += len(goldens)
		pbar.set_postfix_str("Train Acc : %d/%d = %.4f%%" % (good_hit , tota_hit , 100 * good_hit / tota_hit))

	valid_res = valid(net , data[C.valid_data] , epoch_num = epoch_num)

	logger.log("--------------------------------------------------------------------")
	logger.log("Epoch %d ended." % (epoch_num + 1))
	logger.log("Train Acc : %d/%d = %.4f%%" % (good_hit , tota_hit , 100 * good_hit / tota_hit))
	logger.log("Valid Acc : %d/%d = %.4f%%" % (valid_res[0] , valid_res[1] , 100 * valid_res[0] / valid_res[1]))




#---------------------------------------------------------------------------------------------------

#train_result = trainer.train(load_best_model = True)
#logger.log("train: {0}".format(train_result))
#
#print ("Training done. Now testing.")
#tester = Tester(
#	data 	= test_data , 
#	model 	= net , 
#	metrics = AccuracyMetric(pred = "pred" , target = "label") ,
#	device 	= C.gpus , 
#)
#test_result = tester.test()
#logger.log("test: {0}".format(test_result))

#---------------------------------------------------------------------------------------------------
