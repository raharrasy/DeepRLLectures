import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from Networks import Actor, Critic
from torch.autograd import Variable

import gym

class DDPG(object):
	def __init__(self, inputSize , outputSpace, actorLearningRate=1e-4, criticLearningRate=1e-3,
		batchSize = 64, discountRate=0.99, numHiddenUnits = 80,
		polyakAveragingWeight = 1e-3):

		self.inputSize = inputSize
		self.outputSpace = outputSpace
		self.actorLearningRate = actorLearningRate
		self.criticLearningRate = criticLearningRate
		self.batchSize = batchSize
		self.discountRate = discountRate
		self.numHiddenUnits = numHiddenUnits
		self.polyakAveragingWeight = polyakAveragingWeight

		self.actor = Actor(self.inputSize, self.numHiddenUnits, self.outputSpace)
		self.critic = Critic(self.inputSize, self.numHiddenUnits, self.outputSpace)
		self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actorLearningRate)
		self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.criticLearningRate)

		self.targetActor = Actor(self.inputSize, self.numHiddenUnits, self.outputSpace)
		self.targetCritic = Critic(self.inputSize, self.numHiddenUnits, self.outputSpace)


		self.hard_update()

	def select_action(self, state, explorationNoise=None):
		with torch.no_grad():
			self.actor.eval()
			action = self.actor((Variable(state)))
			self.actor.train()
			action = action.data
			if explorationNoise !=None:
				action += torch.Tensor(explorationNoise.noise())
		return action.clamp(-1, 1)

	def soft_update(self):
		tau = self.polyakAveragingWeight
		for target_param, param in zip(self.targetActor.parameters(), self.actor.parameters()):
			target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

		for target_param, param in zip(self.targetCritic.parameters(), self.critic.parameters()):
			target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

	def hard_update(self):
		for target_param, param in zip(self.targetActor.parameters(), self.actor.parameters()):
			target_param.data.copy_(param.data)

		for target_param, param in zip(self.targetCritic.parameters(), self.critic.parameters()):
			target_param.data.copy_(param.data)

	def update_parameters(self, batch):
		states = Variable(torch.cat(batch.state))
		action = Variable(torch.cat(batch.action))
		reward = Variable(torch.cat(batch.reward))
		mask = Variable(torch.cat(batch.mask))
		next_state = Variable(torch.cat(batch.next_state))
		
		with torch.no_grad():
			next_actions = self.targetActor(next_state)
			next_state_action_values = self.targetCritic(next_state, next_actions)
			reward = reward.unsqueeze(1)
			mask = mask.unsqueeze(1)
			expected_state_action = reward + (self.discountRate * mask * next_state_action_values)


		self.critic_optim.zero_grad()

		state_action = self.critic((states), (action))

		value_loss = F.mse_loss(state_action, expected_state_action)
		value_loss.backward()
		self.critic_optim.step()

		self.critic_optim.zero_grad()
		self.actor_optim.zero_grad()

		policy_loss = -self.critic((states),self.actor((states)))

		policy_loss = policy_loss.mean()
		policy_loss.backward()
		self.actor_optim.step()

		#self.soft_update()

		return value_loss.item(), policy_loss.item()


	def save_actor(self, path_name):
		torch.save(self.actor.state_dict(), path_name)

	def load_actor(self, path_name):
		self.actor.load_state_dict(torch.load(path_name))
