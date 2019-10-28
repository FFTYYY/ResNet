from .resnet_cifar import resnet56_cifar
import torch as tc
import torch.nn as nn

class ResNet_56_3p(nn.Module):
	def __init__(self , num_class = 10  ,input_size = [32,32]):
		super().__init__()

		assert input_size == [32,32]

		self.resnet = resnet56_cifar(num_classes = num_class)
	def choose_kwargs():
		return []

	def forward(self , s):
		res = self.resnet(s)

		return {
			"pred" : res , 
		}
