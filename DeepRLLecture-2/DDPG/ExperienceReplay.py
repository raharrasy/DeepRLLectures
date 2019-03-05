from Buffer import Buffer
import numpy as np
from collections import namedtuple

Transition = namedtuple(
    'Transition', ('state', 'action', 'mask', 'next_state', 'reward'))

class ExperienceReplay(object): 
    def __init__(self,maxSize):
        self.maxSize = maxSize
        self.buffer = Buffer(self.maxSize)
        self.curSize = 0

    def addExperience(self, *experience):
        self.buffer.insert(Transition(*experience))
        self.curSize = min(self.curSize+1,self.maxSize)

    def sample(self, samplesAmount):
        sampledPoints = np.random.choice(self.curSize, samplesAmount, replace=False).tolist()
        expList = []
        for a in sampledPoints :
                expList.append(self.buffer.getItem(a))

        return expList