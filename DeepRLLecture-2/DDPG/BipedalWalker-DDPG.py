
import gym
from DDPGController import DDPG
from ExperienceReplay import ExperienceReplay, Transition
from OUNoise import OUNoise
import torch

#env = gym.make('HalfCheetah-v2')
env = gym.make('BipedalWalker-v2')
#controller = DDPG(17, env.action_space)
controller = DDPG(24, env.action_space)

controller.save_actor("model_params/0")
expReplaySize = int(1e6)
experienceReplay = ExperienceReplay(expReplaySize)
#noise = OUNoise(6)
noise = OUNoise(4)
batchSize = 64
updateFrequencies = 32
actionCounter = 0

for i_episode in range(2000):
	observation = env.reset()
	done = False
	total = 0
	counter = 0

	if (i_episode+1)%100 == 0:
		controller.save_actor("model_params/"+str((i_episode+1)//100))

	while not done:
		if i_episode%100 < 10:
			env.render()
		action = controller.select_action(torch.Tensor([observation]), noise)
		newObservation, reward, done, info = env.step(action[0])
		actionCounter += 1
		total += reward
		experienceReplay.addExperience(torch.Tensor([observation]), action, torch.Tensor([int(not done)]), torch.Tensor([newObservation]), torch.Tensor([reward]))
		if experienceReplay.curSize >= batchSize :
			samples = experienceReplay.sample(batchSize)
			batch = Transition(*zip(*samples))
			valueLoss, policyLoss = controller.update_parameters(batch)
			if actionCounter % updateFrequencies == 0:
				controller.hard_update()

		observation = newObservation
		counter += 1
		if done:
			break

	print("Episode {} finished after {} timesteps with reward {}".format(i_episode, counter, total))
