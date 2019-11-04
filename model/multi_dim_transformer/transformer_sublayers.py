import sys
import fastNLP
import torch as tc
from torch import nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import pdb

def Attention(Q, K, V):
	bs,h,n,d = Q.size()

	y = Q.matmul(K.transpose(-1, -2)) / math.sqrt(d)

	y = F.softmax(y, dim = -1)

	y = y.matmul(V)

	return y

def Attention_GDN(Q, K, V):
	'''
		Q: bs,h,n,d
		seq_mask: bs,n,1
	'''

	bs,h,n,d = Q.size()
	b = bs * h
	m = K.size(-2)

	beta , lamb , X = Q , K , V

	X 	 = X   .view(b , m , d)
	lamb = lamb.view(b , m , d)
	beta = beta.view(b , n , d)


	#---GDNN---
	lamb = lamb * (d ** 0.25)
	#---GDNN---

	#归一化: lambda[b,:,k] , beta[b,i,:]
	lamb = F.softmax(lamb , dim = 1)
	beta = F.softmax(beta , dim = 2)

	lamb = lamb.transpose(1,2)  #(b,d,n)

	X_lamb = lamb.matmul(X)  #(b,d,d)
	Z = beta.matmul(X_lamb)  #(b,n,d)

	return Z

class MultiHeadAttention(nn.Module):
	def __init__(self , h = 8 , d_model = 512 , n_persis = 512 , drop_p = 0.0):
		super(MultiHeadAttention, self).__init__()

		dk = d_model // h

		self.WQ = nn.Linear(d_model, d_model, bias = False)
		self.WK = nn.Linear(d_model, d_model, bias = False)
		self.WV = nn.Linear(d_model, d_model, bias = False)

		self.WO = nn.Linear(d_model, d_model, bias = False)

		self.MK = nn.Parameter( tc.zeros(1 , h , n_persis , dk) )
		self.MV = nn.Parameter( tc.zeros(1 , h , n_persis , dk) )

		self.drop = nn.Dropout(drop_p)
	#	self.drop2 = nn.Dropout(dorp_p)

		#-----hyper params-----
		self.n_persis = n_persis
		self.dk = dk
		self.h = h

		self.reset_parameters()

	def reset_parameters_persis(self):
		nn.init.normal_(self.MK.data , mean = 0. , std = 1 / self.dk)
		self.MK.data = self.MK.data * (self.dk ** 0.5)

		nn.init.normal_(self.MV.data , mean = 0. , std = 1 / self.n_persis)
		self.MV.data = self.MV.data * (self.n_persis ** 0.5)

	def reset_parameters(self):
		#nn.init.uniform_(self.WQ.weight.data , - (d_model**-0.5) , (d_model**-0.5))
		#nn.init.uniform_(self.WK.weight.data , - (d_model**-0.5) , (d_model**-0.5))
		#nn.init.uniform_(self.WV.weight.data , - (d_model**-0.5) , (d_model**-0.5))
		#nn.init.uniform_(self.WO.weight.data , - (d_model**-0.5) , (d_model**-0.5))

		self.reset_parameters_persis()

	def forward(self, Q, K, V):
		'''
			Q: bs , n , d
			mas : bs , n , 1
			mas_att : bs , n , n
		'''
		#pdb.set_trace()

		bs , n , d = Q.size()
		h = self.h

		Q , K , V = self.WQ(Q) , self.WK(K) , self.WV(V)

		Q = Q.view(bs,n,h,self.dk).transpose(1,2).contiguous()	#(bs,h,n,d)
		K = K.view(bs,n,h,self.dk).transpose(1,2).contiguous()	#(bs,h,n,d)
		V = V.view(bs,n,h,self.dk).transpose(1,2).contiguous()	#(bs,h,n,d)

		K = tc.cat([K, self.MK.expand(bs,h,self.n_persis,self.dk)] , dim = -2)
		V = tc.cat([V, self.MV.expand(bs,h,self.n_persis,self.dk)] , dim = -2)

		#y = Attention(Q , K , V, mas_att , None)
		y = Attention_GDN(Q , K , V)

		y = y.view(bs,h,n,self.dk).transpose(1,2).contiguous().view(bs,n,h*self.dk)

		y = self.WO(y)

		return y


class FFN(nn.Module):
	def __init__(self, d_model = 512, d_hid = 512, drop_p = 0.0):
		super(FFN, self).__init__()

		self.d_hid = d_hid
		self.L1 = nn.Linear(d_model , d_hid , bias = True)
		self.L2 = nn.Linear(d_hid , d_model , bias = True)
		self.drop = nn.Dropout(drop_p)		
		#self.reset_parameters()

	def reset_parameters(self):
		nn.init.xavier_normal_(self.L1.weight.data)
		nn.init.xavier_normal_(self.L2.weight.data)
		self.L1.bias.data.fill_(0)
		self.L2.bias.data.fill_(0)

	def forward(self, x , mas):
		x = self.drop(F.relu(self.L1(x)))
		x = self.L2(x)
		return x


class Encoder_Layer(nn.Module):
	def __init__(self , d_model = 512 , d_hid = 512 , n_persis = 512 , h = 8 , drop_p = 0.0):
		super(Encoder_Layer, self).__init__()

		self.multi_att = MultiHeadAttention(h = h , d_model = d_model , n_persis = n_persis , drop_p = drop_p)
		self.layernorm_1 = nn.LayerNorm([d_model])
		self.drop_1 = nn.Dropout(drop_p)

		self.ffn = FFN(d_model = d_model , d_hid = d_hid , drop_p = drop_p)
		self.layernorm_2 = nn.LayerNorm([d_model])
		self.drop_2 = nn.Dropout(drop_p)

	def forward(self, x):

		out1 = self.multi_att(x , x , x)
		x = self.layernorm_1(x + self.drop_1(out1))

		#out2 = self.ffn(x , mas)
		#x = self.layernorm_2(x + self.drop_2(out2 * mas))
		#x *= mas

		return x

class Encoder(nn.Module):
	def __init__(self , num_layers = 4 , d_model = 512 , d_hid = 512 , n_persis = 512 , h = 8 , drop_p = 0.0):
		super(Encoder, self).__init__()

		self.enc_layer = nn.ModuleList([Encoder_Layer(d_model,d_hid,n_persis,h,drop_p) for _ in range(num_layers)])

		self.conv_layer = nn.ModuleList([nn.Conv2d(d_model,d_model,3,padding = 1) for _ in range(num_layers)])
		self.conv_ln = nn.LayerNorm(d_model)

		#-----hyper params-----
		self.d_model = d_model
		self.num_layers = num_layers

	def forward(self, x , input_size = [32,32]):
		bs , l1 , l2 , d_model = x.size(0) , input_size[0] , input_size[1] , x.size(2)

		for i in range(self.num_layers):
			enc_layer = self.enc_layer[i]
			conv_layer = self.conv_layer[i]

			x = enc_layer(x) #(bs , len , d_model)

			x = x.permute(0,2,1).contiguous().view(bs,d_model,l1,l2) #(bs , d_model , len)
			x = F.relu(conv_layer(x)) + x
			x = self.conv_ln( x.view(bs,d_model,l1*l2).permute(0,2,1).contiguous() ) #(bs , len , d_model)

		return x
