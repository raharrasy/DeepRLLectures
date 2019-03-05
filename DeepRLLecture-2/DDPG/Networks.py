import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class Actor(nn.Module):
	def __init__(self, inputSize, hiddenLayerSize, actionSpace):

		super(Actor, self).__init__()
		self. inputSize = inputSize
		self.actionSpaceSize = actionSpace.shape[0]

		self.linear1 = nn.Linear(inputSize, hiddenLayerSize)
		self.layerNorm = nn.LayerNorm(hiddenLayerSize)


		self.linear1.weight = torch.nn.Parameter(2.0/math.sqrt(inputSize) * torch.rand(self.linear1.weight.shape[0], 
			self.linear1.weight.shape[1]) - 1.0/math.sqrt(inputSize))
		self.linear1.bias = torch.nn.Parameter(2.0/math.sqrt(inputSize) * torch.rand(self.linear1.bias.shape[0]) - 1.0/math.sqrt(inputSize))

		self.linear2 = nn.Linear(hiddenLayerSize, hiddenLayerSize)
		self.layerNorm2 = nn.LayerNorm(hiddenLayerSize)

		self.linear2.weight = torch.nn.Parameter(2.0/math.sqrt(inputSize) * torch.rand(self.linear2.weight.shape[0], 
			self.linear2.weight.shape[1]) - 1.0/math.sqrt(inputSize))
		self.linear2.bias = torch.nn.Parameter(2.0/math.sqrt(inputSize) * torch.rand(self.linear2.bias.shape[0]) - 1.0/math.sqrt(inputSize))

		self.acts = nn.Linear(hiddenLayerSize, self.actionSpaceSize)
		self.acts.weight = torch.nn.Parameter(6e-3 * torch.rand(self.acts.weight.shape[0], 
			self.acts.weight.shape[1]) - 3e-3)
		self.acts.bias = torch.nn.Parameter(6e-3 * torch.rand(self.acts.bias.shape[0]) - 3e-3)

	def forward(self, inputs) :
		out = self.linear1(inputs)
		out = self.layerNorm(out)
		out = F.relu(out)
		out = self.linear2(out)
		out = self.layerNorm2(out)
		out = F.relu(out)
		out = torch.tanh(self.acts(out))

		return out


class Critic(nn.Module):
	def __init__(self, inputSize, hiddenLayerSize, actionSpace):

		super(Critic, self).__init__()
		self. inputSize = inputSize
		self.actionSpaceSize = actionSpace.shape[0]

		self.linear1 = nn.Linear(inputSize, hiddenLayerSize)
		self.layerNorm = nn.LayerNorm(hiddenLayerSize)

		self.linear1.weight = torch.nn.Parameter(2.0/math.sqrt(inputSize) * torch.rand(self.linear1.weight.shape[0], 
			self.linear1.weight.shape[1]) - 1.0/math.sqrt(inputSize))
		self.linear1.bias = torch.nn.Parameter(2.0/math.sqrt(inputSize) * torch.rand(self.linear1.bias.shape[0]) - 1.0/math.sqrt(inputSize))

		self.linear2 = nn.Linear(hiddenLayerSize+self.actionSpaceSize, hiddenLayerSize)
		self.layerNorm2 = nn.LayerNorm(hiddenLayerSize)

		self.linear2.weight = torch.nn.Parameter(2.0/math.sqrt(inputSize) * torch.rand(self.linear2.weight.shape[0], 
			self.linear2.weight.shape[1]) - 1.0/math.sqrt(inputSize))
		self.linear2.bias = torch.nn.Parameter(2.0/math.sqrt(inputSize) * torch.rand(self.linear2.bias.shape[0]) - 1.0/math.sqrt(inputSize))

		self.acts = nn.Linear(hiddenLayerSize, 1)
		self.acts.weight = torch.nn.Parameter(6e-3 * torch.rand(self.acts.weight.shape[0], 
			self.acts.weight.shape[1]) - 3e-3)
		self.acts.bias = torch.nn.Parameter(6e-3 * torch.rand(self.acts.bias.shape[0]) - 3e-3)

	def forward(self, inputs,actions) :
		out = self.linear1(inputs)
		out = self.layerNorm(out)
		out = F.relu(out)
		out = torch.cat((out,actions),1)
		out = self.linear2(out)
		out = self.layerNorm2(out)
		out = F.relu(out)
		out = self.acts(out)

		return out




