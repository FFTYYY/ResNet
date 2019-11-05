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
import torchvision
import torchvision.transforms as transforms

from model.multi_dim_transformer import Model as MD_Transformer
from model.resnet import Model as ResNet
from model.third_party import ResNet_56_3p
from config import C , logger
from dataloader_cifar10 import n_crop_test as n_crop_test_cifar_10

import PIL

import pdb


#---------------------------------------------------------------------------------------------------
#Get model

with open(os.path.join(C.model_path , C.model_save) , "rb") as fil:
	net = pickle.load( fil ).cuda()

logger.log ("Load network done.")

#---------------------------------------------------------------------------------------------------
#Get data
transform_vis  = transforms.Compose([
	transforms.Resize( (32,32) ),
	transforms.ToTensor(),
	transforms.Normalize(mean = [0.491 , 0.482 , 0.447] , std = [0.247 , 0.243 , 0.262]),
])

if C.file == "":
	C.file = "./visualize_data/1.jpg"

pict = PIL.Image.open(C.file)

x = transform_vis(pict).cuda()

#---------------------------------------------------------------------------------------------------

classes = [
	"飞机",
	"汽车",
	"鸟",
	"猫猫",
	"鹿",
	"狗狗",
	"青蛙",
	"马",
	"船",
	"卡车",
]

y = net(x.unsqueeze(0))["pred"][0]
y = tc.softmax(y , dim = -1)
c = int(tc.max(y , -1)[1])


print ("是 %s 哒！（信心：%.2f%%）" % (classes[c] , y[c] * 100))
print ("\n信心：")

for x in range(len(classes)):
	print ("%s\t: %.2f%%" % (classes[x] , y[x] * 100))
