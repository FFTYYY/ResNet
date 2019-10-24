import torch as tc
from torch import nn
import torch.nn.functional as F
import fastNLP
import math
import pdb

class ResNetLayer_1(nn.Module):
	def __init__(self , n_in_channels , n_out_channels , sub_sampling = False):
		'''
			subsampling : stride = 2 to decrease the feature map
		'''

		super().__init__()

		self.conv1 = nn.Conv2d(n_in_channels , n_in_channels , 
				kernel_size = 3 , padding = 1 , stride = 1)
		self.bn1 = nn.BatchNorm2d(n_in_channels)
		self.conv2 = nn.Conv2d(n_in_channels , n_out_channels , 
				kernel_size = 3 , padding = 1 , stride = 2 if sub_sampling else 1)
		self.bn2 = nn.BatchNorm2d(n_out_channels)

		self.ln = None
		if n_in_channels != n_out_channels:
			self.ln = nn.Linear(n_in_channels , n_out_channels)

	def forward(self , x):
		old_x = x
		x = F.relu(self.bn1(self.conv1(x)))
		x = self.bn2(self.conv2(x))

		if self.ln is not None:
			s = list(x.size())
			old_x = old_x[:s[0] , :s[1] , :s[2] , :s[3]]
			old_x = self.ln(old_x.permute(0,2,3,1)).permute(0,3,1,2)

		x = F.relu(x + old_x)
		return x