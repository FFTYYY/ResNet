import sys
from fastNLP import Optimizer
import torch
import torch.nn as nn
import numpy as np

class ScheduledOptim():
	'''A simple wrapper class for learning rate scheduling'''

	def __init__(self, model_params , d_model, n_warmup_steps , init_steps  , step_size):
		self.init_lr = np.power(d_model, -0.5)
		self._optimizer = torch.optim.Adam(params = model_params , lr = self.init_lr , betas = (0.9,0.98))
		self.n_warmup_steps = n_warmup_steps
		self.n_current_steps = init_steps
		self.step_size = step_size

	def step(self):
		self._update_learning_rate()
		self._optimizer.step()

	def zero_grad(self):
		self._optimizer.zero_grad()

	def _get_lr_scale(self):
		return np.min([
			np.power(self.n_current_steps, -0.5),
			np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

	def _update_learning_rate(self):
		''' Learning rate scheduling per step '''

		self.n_current_steps += self.step_size
		lr = self.init_lr * self._get_lr_scale()

		for param_group in self._optimizer.param_groups:
			param_group['lr'] = lr


class MyAdam(Optimizer):
	def __init__(self,  d_model , n_warmup_steps , init_steps = 0 , model_params = None , step_size = 1):

		super(MyAdam, self).__init__(
				model_params , 
				d_model = d_model , 
				n_warmup_steps = n_warmup_steps , 
				init_steps = init_steps , 
				step_size = step_size , 
			)

	def construct_from_pytorch(self, model_params):
		if self.model_params is None:
			return ScheduledOptim(model_params, **self.settings)
		else:
			return ScheduledOptim(self.model_params, **self.settings)
