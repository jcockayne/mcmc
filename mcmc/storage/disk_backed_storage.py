import numpy as np
import os


class DiskBackedStorage(object):
    def __init__(self, store_shape, disk_path):
        self.__store_shape__ = store_shape
        self.__array__ = np.empty(store_shape)
        self.__max_rows__ = store_shape[0]
        self.__disk_path__ = disk_path
        self.__shape_offset__ = 0
        self.__last_index_written__ = -1 # nothing is there yet
        if os.path.exists(self.__disk_path__):
            raise Exception('File {} already exists'.format(self.__disk_path__))
        self.__f__ = open(self.__disk_path__, 'a+b')

    def __check_and_adjust_key__(self, key):
        # need to determine if I'm at the end of my array
        assert type(key) is int

        key -= self.__max_rows__ * self.__shape_offset__
        assert key > self.__last_index_written__

        return key

    def set_item(self, index, value):
        row_key = self.__check_and_adjust_key__(index)

        if row_key >= self.__max_rows__:
            self.__flush_to_disk__()
            row_key = self.__check_and_adjust_key__(index)

        self.__last_index_written__ = row_key

        self.__array__[row_key, :] = value

    def __flush_to_disk__(self):
        # handle case when nothing is in the cache
        if self.__last_index_written__ == -1:
            return
        np.save(self.__f__, self.__array__[:self.__last_index_written__, :])
        self.__array__ = np.empty(self.__store_shape__)
        self.__shape_offset__ += 1
        self.__last_index_written__ = -1

    @property
    def disk_path(self):
        return self.__disk_path__

    def finish(self):
        self.__flush_to_disk__()
        self.__f__.close()

