import numpy as np
class ArrayStorage(object):
    def __init__(self, arr_size):
        self.__array__ = np.empty(arr_size)
        self.__current_index__ = 0

    def add_sample(self, item):
        self.__array__[self.__current_index__, :] = item
        self.__current_index__ += 1

    @property
    def array(self):
        return self.__array__
