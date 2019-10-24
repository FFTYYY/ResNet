import os , sys

import fastNLP
from fastNLP import Vocabulary , DataSet , Instance , Tester
from fastNLP import Trainer , AccuracyMetric , CrossEntropyLoss , Adam
import torch.nn as nn
import torch as tc
from torch.optim import Adadelta
import numpy as np
import random
import pickle
from tqdm import tqdm

from config import C , logger
from model.multi_dim_transformer import Model as MD_Transformer
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

logger.log ("Data load done.")
logger.log ("train size = %d test size = %d" % (len(data["train"]) , len(data["test"])))
#---------------------------------------------------------------------------------------------------
#Get model

net = tc.load( os.path.join(C.model_save , C.model , C.model_load) ).cuda( C.gpus[0] )

logger.log ("Load network done.")

#---------------------------------------------------------------------------------------------------
#Train & Test

good_hit = 0
for data in tqdm(test_data , ncols = 70):
	s = data["s"]
	gold = int(data["label"])

	y = net( s.unsqueeze(0).cuda( C.gpus[0] ) )["pred"][0]

	y = int(tc.max(y , -1)[1])
	if y == gold:
		good_hit += 1
print ("Test Accurancy : %d/%d = %.2f%%" % (good_hit , len(test_data) , 100 * good_hit / len(test_data)))

#---------------------------------------------------------------------------------------------------
