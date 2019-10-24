import sys
import fastNLP
import torch as tc
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import pdb
from .pos_emb import *
from .transformer_sublayers import *

Position_Embedding = PE_1

class Model(nn.Module):
	def __init__(self, num_class , input_size = [32,32] ,num_layers = 4 , 
			d_model = 512 , d_hid = 512 , d_posenc = 512 , n_persis = 512 , 
			h = 8 , drop_p = 0.0 , max_k = 4):
		super().__init__()

		self.input_conv = nn.Conv2d(3 , d_model , 3 , padding = 1)
		self.bn1 		= nn.BatchNorm2d(d_model)

		self.pe 		= Position_Embedding(d_model , input_size , d_posenc)
		self.pe_drop 	= nn.Dropout(drop_p)
		self.enc 		= Encoder(num_layers , d_model , d_hid , n_persis , h , drop_p)
		
		self.l1 		= nn.Linear(d_model , num_class)
		self.conv1 		= nn.Conv2d(d_model,d_model,3,padding = 1)

		#-----hyper params-----
		self.d_model = d_model
		self.input_size = input_size
		self.max_k = max_k

	def choose_kwargs():
		return ["num_layers" , "d_model" , "d_hid" , "n_persis" , "h" , "drop_p" , "d_posenc" , "max_k"]

	def forward(self , s):

		siz = s.size(0)
		len_1 , len_2 = self.input_size

		s = self.bn1(F.relu(self.input_conv(s)))
		matrix = s.permute(0,2,3,1).contiguous()

		matrix = matrix.view(siz , len_1 , len_2 , self.d_model)	# (siz , l1 , l2 , 2 * emb)
		matrix = self.pe(matrix)
		matrix = matrix.view(siz , len_1 * len_2 , self.d_model)	# (siz , l1 * l2 , d)

		x = self.pe_drop(matrix)

		x = self.enc(x , self.input_size)

		#-----conv-----
		x = x.view(siz , len_1 , len_2 , self.d_model).permute(0,3,1,2).contiguous()

		# (siz , d , l1 , l2)
		x = self.conv1(x)

		# (siz , d , l1 , k)
		x = x.gather(-1,tc.topk(x,self.max_k,dim = -1)[1].sort(dim = -1)[0])

		# (siz , d , k , k)
		x = x.gather(-2,tc.topk(x,self.max_k,dim = -2)[1].sort(dim = -2)[0])

		x = F.relu(x)

		# (siz , d , k * k)
		y = x.view(siz , self.d_model , -1)

		#y = self.con_ln(y).squeeze(-1)

		y = tc.sum(y , dim = -1)

		#-----conv-----

		y = self.l1(y)

		return {
			"pred": y,
			#"mat": matrix,
		}
