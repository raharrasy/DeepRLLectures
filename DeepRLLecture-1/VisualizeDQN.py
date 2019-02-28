import gym
import torch
import random
from torch.autograd import Variable
from ExperienceReplay import ExperienceReplay
from ValueNetwork import ValueNetwork
import torch.optim as optim
import torch.nn as nn
import argparse

def compute_val(value_network, obs):
	var_obs = Variable(obs)
	output_qs = value_network((var_obs))
	return output_qs 

def hard_copy(targetValueNetwork, valueNetwork):
	for target_param, param in zip(targetValueNetwork.parameters(), valueNetwork.parameters()):
					target_param.data.copy_(param.data)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('idUsed', type=int,
                    help='an integer for params')
args = parser.parse_args()
	
env = gym.make('CartPole-v0')
numEpisodes  = 4000
total_actions = 0
learning_rate = 1e-3
max_grads = 1.0
batch_size = 128
gamma = 0.999
copy_freq = 2000

value_network = ValueNetwork(4,[16,16,4],2)
value_network.load_state_dict(torch.load("DQNParams/{}".format(str(args.idUsed))))

for episode in range(numEpisodes):
	obs = env.reset()
	done = False
	epsilon = 0.05
		
	total_reward = 0.0
	env.render()
	while not done:
		obs_tensor = torch.Tensor(obs).unsqueeze(0)
		tens = compute_val(value_network, obs_tensor)
		act = torch.max(tens, dim=1)[1].item()

		next_obs, rewards, done, info = env.step(act)
		total_reward += rewards
		total_actions += 1
		env.render()

			
		obs = next_obs
	print(total_reward)
