
import gym
from DDPGController import DDPG
from ExperienceReplay import ExperienceReplay, Transition
from OUNoise import OUNoise
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--experimentId", type=int, default=0)
args = parser.parse_args()

#env = gym.make('HalfCheetah-v2')
env = gym.make('BipedalWalker-v2')
#controller = DDPG(17, env.action_space)
controller = DDPG(24, env.action_space)

controller.load_actor("model_params/"+str(args.experimentId))
noise = OUNoise(4)
#noise = OUNoise(6)
batchSize = 64
updateFrequencies = 32
actionCounter = 0

for i_episode in range(2000):
	observation = env.reset()
	done = False
	total = 0
	counter = 0

	while not done:
		env.render()
		action = controller.select_action(torch.Tensor([observation]),noise)
		newObservation, reward, done, info = env.step(action[0])
		actionCounter += 1
		total += reward

		observation = newObservation
		counter += 1
		if done:
			break

	print("Episode {} finished after {} timesteps with reward {}".format(i_episode, counter, total))
