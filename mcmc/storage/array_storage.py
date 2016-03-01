import numpy as np
class ArrayStorage(object):
    def __init__(self, arr_size):
        self.__array__ = np.empty(arr_size)

    def set_item(self, ix, item):
        self.__array__[ix, :] = item

    @property
    def array(self):
        return self.__array__
