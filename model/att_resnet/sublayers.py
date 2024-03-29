import torch as tc
from torch import nn
import torch.nn.functional as F
import fastNLP
import math
import pdb
from ..multi_dim_transformer.transformer_sublayers import MultiHeadAttention as _MultiHeadAttention

#adapter
class MultiHeadAttention(nn.Module):
	def __init__(self , h = 8 , d_model = 512 , n_persis = 512 , drop_p = 0.0):
		super().__init__()

		self.att = _MultiHeadAttention(h,d_model,n_persis,drop_p)

	def forward(self , x):
		'''
			x: (bs , fnum , fmap , fmap)
		'''
		bs , fnum , fmap , fmap = x.size()
		x = x.view(bs , fnum , fmap * fmap).transpose(1 , 2).contiguous()

		x = self.att(x , x , x)

		x = x.transpose(1 , 2).contiguous().view(bs , fnum , fmap , fmap)

		return x


class ResNetLayer_2(nn.Module):
	def __init__(self , n_in_channels , n_out_channels , sub_sampling = False , drop_p = 0.0):
		'''
			subsampling : stride = 2 to decrease the feature map
		'''

		super().__init__()

		self.attn = MultiHeadAttention(h = 4 , d_model = n_in_channels , n_persis = n_in_channels , drop_p = drop_p)

		self.conv1 = nn.Conv2d(n_in_channels , n_in_channels , 
				kernel_size = 3 , padding = 1 , stride = 1)
		self.bn1 = nn.BatchNorm2d(n_in_channels)
		self.conv2 = nn.Conv2d(n_in_channels , n_out_channels , 
				kernel_size = 3 , padding = 1 , stride = 2 if sub_sampling else 1)
		self.bn2 = nn.BatchNorm2d(n_out_channels)

		self.proj = None
		if n_in_channels != n_out_channels:
			self.proj = nn.Conv2d(n_in_channels , n_out_channels , 
				kernel_size = 1 , padding = 0 , stride = 2 if sub_sampling else 1)

		self.dropout = nn.Dropout(drop_p)

		self.reset_parameters()

	def reset_parameters(self):
		#nn.init.xavier_normal_(self.conv1.weight.data , gain = 1)
		#nn.init.xavier_normal_(self.conv2.weight.data , gain = 1)
		#if self.proj is not None:
		#	nn.init.xavier_normal_(self.proj.weight.data , gain = 1)
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()


	def forward(self , x):
		old_x = x
		x = F.relu(self.attn(x))
		x = F.relu(self.bn1(self.conv1(x)))
		x = self.bn2(self.conv2(x))

		if self.proj is not None:
			old_x = self.proj(old_x)

		x = self.dropout(F.relu(x + old_x))

		return x