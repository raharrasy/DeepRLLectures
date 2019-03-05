import numpy as np


# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class GaussianNoise:

    def __init__(self, action_dimension, scale=1.0, mu=0 , sigma=0.1):
        self.action_dimension = action_dimension
        self.mu = mu
        self.sigma = sigma

    def noise(self):
        return self.sigma * np.random.randn(self.action_dimension) + self.mu