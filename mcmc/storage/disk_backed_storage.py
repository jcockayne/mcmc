import numpy as np
import os


class DiskBackedStorage(object):
    def __init__(self, store_shape, disk_path):
        self.__store_shape__ = store_shape
        self.__array__ = np.empty(store_shape)
        self.__max_rows__ = store_shape[0]
        self.__disk_path__ = disk_path
        self.__shape_offset__ = 0

    def __check_and_adjust_key__(self, key):
        # need to determine if I'm at the end of my array
        assert type(key) is tuple and len(key) == 2, 'DiskBackedStorage only supports a 2D array.'
        row_key, col_key = key
        assert type(row_key) is int, 'DiskBackedStorage only supports assignment one row at a time.'

        row_key -= self.__max_rows__ * self.__shape_offset__

        return row_key, col_key

    def __setitem__(self, key, value):
        row_key, col_key = self.__check_and_adjust_key__(key)

        if row_key >= self.__max_rows__:
            self.__flush_to_disk__()
            row_key, col_key = self.__check_and_adjust_key__(key)

        self.__array__[row_key, col_key] = value

    def __getitem__(self, item):
        row_key, col_key = self.__check_and_adjust_key__(item)

        assert self.__store_shape__ > row_key >= 0, \
            'DiskBackedStorage cannot access elements outside of current memory block.'

        return self.__array__[row_key, col_key]

    def __flush_to_disk__(self):
        if not os.path.exists(self.__disk_path__):
            f = open(self.__disk_path__, 'w')
        else:
            f = open(self.__disk_path__, 'a')
        np.savetxt(f, self.__array__)
        self.__array__ = np.empty(self.__store_shape__)
        self.__shape_offset__ += 1
