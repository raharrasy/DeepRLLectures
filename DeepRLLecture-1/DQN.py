import gym
import torch
import random
from torch.autograd import Variable
from ExperienceReplay import ExperienceReplay
from ValueNetwork import ValueNetwork
import torch.optim as optim
import torch.nn as nn

def compute_val(value_network, obs):
	var_obs = Variable(obs)
	output_qs = value_network((var_obs))
	return output_qs 

def hard_copy(targetValueNetwork, valueNetwork):
	for target_param, param in zip(targetValueNetwork.parameters(), valueNetwork.parameters()):
					target_param.data.copy_(param.data)

if __name__ == "__main__":
	env = gym.make('CartPole-v0')
	numEpisodes  = 4000
	exp_replay = ExperienceReplay(20000)
	total_actions = 0
	learning_rate = 1e-3
	max_grads = 1.0
	batch_size = 128
	gamma = 0.999
	copy_freq = 2000

	value_network = ValueNetwork(4,[16,16,4],2)
	target_value_network = ValueNetwork(4,[16,16,4],2)
	hard_copy(target_value_network, value_network)

	optimizer = optim.Adam(value_network.parameters(), lr=learning_rate)

	for episode in range(numEpisodes):
		obs = env.reset()
		done = False
		epsilon = 1.0 - (min(1.0, numEpisodes/4000.0) * 0.95)
		
		total_reward = 0.0
		if episode % 1000 == 0:
				env.render()

		if episode % 1000 == 0:
			torch.save(value_network.state_dict(), "DQNParams/{}".format(str(episode//500)))

		while not done:
			obs_tensor = torch.Tensor(obs).unsqueeze(0)
			tens = compute_val(value_network, obs_tensor)
			print(tens)
			act = torch.max(tens, dim=1)[1].item()
			if random.random() < epsilon:
				act = random.randint(0,1)

			next_obs, rewards, done, info = env.step(act)
			total_reward += rewards
			total_actions += 1

			exp_replay.insert((obs , act, rewards, next_obs, int(done)))

			if episode % 1000 == 0:
				env.render()
			
			if total_actions > batch_size:

				stateTensors, actionTensors, rewardTensors, nextStateTensors, masksTensor = exp_replay.sample(batch_size)
				predicted_vals = compute_val(value_network, stateTensors).gather(1, actionTensors)
				target_next_state = torch.max(compute_val(target_value_network, nextStateTensors), dim=1, keepdim=True)[0]
				target_vals = rewardTensors + gamma * target_next_state * (1 - masksTensor)
				target_vals = target_vals.detach()

				optimizer.zero_grad()

				loss_function = nn.MSELoss()
				err = loss_function(predicted_vals, target_vals)
				err.backward()
				for param in value_network.parameters():
					param.grad.data.clamp_(-max_grads, max_grads)
				optimizer.step()


			if total_actions % copy_freq == 0:
				hard_copy(target_value_network, value_network)
			obs = next_obs

		print(episode, total_reward)

	torch.save(value_network.state_dict(), "DQNParams/{}".format(str(10)))
