class RingBuffer:
    def __init__(self, size_max):
        self.max = size_max
        self.data = []

    def append(self, x):
        """append an element at the end of the buffer"""
        self.data.append(x)
        if len(self.data) == self.max:
            self.cur = 0
            self.__class__ = RingBufferFull

    def get(self):
        """ return a list of elements from the oldest to the newest"""
        return self.data


class RingBufferFull:
    def __init__(self, n):
        raise Exception("you should use RingBuffer")

    def append(self, x):
        self.data[self.cur] = x
        self.cur = (self.cur + 1) % self.max

    def get(self):
        return self.data[self.cur:] + self.data[:self.cur]
