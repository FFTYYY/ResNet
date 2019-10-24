import os , sys

import fastNLP
from fastNLP import Vocabulary , DataSet , Instance , Tester
from fastNLP import Trainer , AccuracyMetric , CrossEntropyLoss
import torch.nn as nn
import torch as tc
from torch.optim import Adadelta
import numpy as np
import random
import pickle

from config import C , logger
from model.multi_dim_transformer import Model as MD_Transformer
from model.resnet import Model as ResNet
from dataloader_cifar10 import load_data as load_data_cifar_10
from optim import MyAdam

import pdb

#---------------------------------------------------------------------------------------------------
#Initialize

if C.seed > 0:
	random.seed(C.seed)
	tc.manual_seed(C.seed)
	np.random.seed(C.seed)
	tc.cuda.manual_seed_all(C.seed)

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
}
model = models[C.model]
net = model(num_class = 10 , input_size = [32,32] ,
	**{x : C.__dict__[x] for x in model.choose_kwargs()}
)

logger.log ("Creat network done.")

#---------------------------------------------------------------------------------------------------
#Train & Test

optims = {
	"myadam" : lambda : MyAdam(d_model = C.d_model , n_warmup_steps = 4000 , init_steps = C.init_steps , step_size = C.step_size) ,
	"adam" 	 : lambda : tc.optim.Adam(params = net.parameters() , lr = C.lr) , 
	"sgd" 	 : lambda : tc.optim.SGD (params = net.parameters() , lr = C.lr) , 
}

optim = optims[C.optim]()

trainer = Trainer(
	train_data 	= data["train"] ,
	dev_data 	= data[C.valid_data] ,
	model 		= net , 
	batch_size 	= C.batch_size,
	loss 		= CrossEntropyLoss(pred = "pred" , target = "label"),
	metrics 	= AccuracyMetric(pred = "pred" , target = "label"),
	optimizer 	= optim,
	n_epochs 	= C.n_epochs,
	save_path 	= os.path.join(C.model_save , C.model , "./"),
	device 		= C.gpus , 
	use_tqdm 	= True,
	check_code_level = -1,
)

train_result = trainer.train(load_best_model = True)
logger.log("train: {0}".format(train_result))

print ("Train done. Now testing.")
tester = Tester(
	data 	= test_data , 
	model 	= net , 
	metrics = AccuracyMetric(pred = "pred" , target = "label") ,
	device 	= C.gpus , 
)
test_result = tester.test()
logger.log("test: {0}".format(test_result))

#---------------------------------------------------------------------------------------------------
