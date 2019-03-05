class Buffer(object):
    def __init__(self,maxSize):
        self.storage = [None] * maxSize
        self.pointer = 0
        self.maxSize = maxSize

    def insert(self,experience):
        self.storage[self.pointer] = experience
        self.pointer = (self.pointer+1)%self.maxSize

    def getItem(self,index):
        return self.storage[index]

    def getPointer(self):
        return self.pointer