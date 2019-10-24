#原来的一个文件里代码太多了，因此分开
import sys
import fastNLP
import torch as tc
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import pdb

class PE_1(nn.Module):
	'''
			concat  ( x , y , x-y , x * y , dx-dy , dx * dy , conv(dx,dy) )
	'''

	def __init__(self , d_model = 512 , len_max = [32,32] , d_posenc = 512):
		super().__init__()

		self.pos_emb_1 = nn.Embedding(len_max[0] , d_posenc)
		self.pos_emb_2 = nn.Embedding(len_max[1] , d_posenc)


		self.ln = nn.Linear(d_model + 2 * d_posenc, d_model)

		self.d_posenc = d_posenc


	def reset_parameters(self):

		nn.init.xavier_normal_(self.ln.weight.data)
		self.ln.bias.data.fill_(0)

	def forward(self, x, mask = None):
		# x : (siz , l1 , l2 , 2 * d)
		#siz = x.size(0)
		#emb_s = x.size(3)
		seq_len_1 , seq_len_2 = x.size(1) , x.size(2)

		self.poss_1 = Variable(tc.LongTensor([[x for y in range(seq_len_2)] for x in range(seq_len_1)])).cuda()
		self.poss_2 = Variable(tc.LongTensor([[y for y in range(seq_len_2)] for x in range(seq_len_1)])).cuda()

		pos_em_1 = self.pos_emb_1(self.poss_1)		#(l1,l2,d_pe)
		pos_em_2 = self.pos_emb_2(self.poss_2)
		pos_em = tc.cat([pos_em_1 , pos_em_2] , dim = -1).view(seq_len_1 , seq_len_2, 2 * self.d_posenc)

		pos_em = pos_em.unsqueeze(0).expand(x.size(0) , seq_len_1 , seq_len_2 , 2 * self.d_posenc)

		if mask is not None: 
			pos_em = pos_em * mask.view(pos_em.size()[:-1] + tc.Size([1]))

		x = tc.cat([x , pos_em], dim=-1)
		#x = tc.cat([x , pos_em , conved_pos_em , dx2 , dx3], dim=-1)

		x = (self.ln(x))

		x = F.relu(x)

		return x

"""
class PE_2(nn.Module):
	'''
	'''

	def get_pe(self , n , d_model):
		'''
			from annotated transformer
		'''
		pe = tc.zeros(n, d_model)
		position = tc.arange(0, n).unsqueeze(1).float()
		div_term = tc.exp(tc.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
		pe[:, 0::2] = tc.sin(position * div_term)
		pe[:, 1::2] = tc.cos(position * div_term)
		return pe


	def __init__(self):
		super().__init__()
		self.pe = self.get_pe(seq_len_max[0] , 2*emb_siz).unsqueeze(1)
		self.pe = self.get_pe(seq_len_max[1] , 2*emb_siz).unsqueeze(0) + self.pe
		self.pe = self.pe.unsqueeze(0)

		self.rel_trans = nn.Embedding(2 , 2*emb_siz)
		self.ln = nn.Linear(2*emb_siz , d_model)

		self.rel = tc.triu( tc.ones(1 , seq_len_max[0] , seq_len_max[1]) ).long()

	def forward(self, x, mask = None):
		'''
			x : (bs , n1 , n2 , 2*emb_siz)
		'''

		if self.pe.device != x.device:
			self.pe = self.pe.to(x.device)
			self.rel = self.rel.to(x.device)

		bs , n1 , n2 , x_d = x.size()

		x = x + self.pe[:,:n1,:n2,:]

		#print (x.size())
		#pdb.set_trace()

		x = x + self.rel_trans(self.rel[:,:n1,:n2])

		x = tc.relu(self.ln(x))
		if mask is not None: 
			x *= mask.view(x.size()[:-1] + tc.Size([1]))


		return x
"""