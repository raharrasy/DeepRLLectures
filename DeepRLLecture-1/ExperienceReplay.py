import numpy as np
import torch

class ExperienceReplay(object):
	def __init__(self, maxSize):
		self.maxSize = maxSize
		self.nextObj = 0
		self.size = 0
		self.storage = [None] * self.maxSize

	def insert(self, data):
		self.storage[self.nextObj] = data
		self.nextObj = (self.nextObj + 1) % self.maxSize
		self.size = min(self.size+1, self.maxSize)

	def sample(self, batchSize) :
		if self.size >= batchSize:
			takenIdxs = np.random.choice(self.size, batchSize, replace=True)
			samples = [self.storage[idx] for idx in takenIdxs]

			state , action, reward, nextState, masks = zip(*samples)
			stateTensors = torch.FloatTensor(state)
			actionTensors = torch.LongTensor(action).view(-1,1)
			rewardTensors = torch.FloatTensor(reward).view(-1,1)
			nextStateTensors = torch.FloatTensor(nextState)
			masksTensor = torch.FloatTensor(masks).view(-1,1)

			return stateTensors, actionTensors, rewardTensors, nextStateTensors, masksTensor
		return None


