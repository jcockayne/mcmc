import numpy as np
import logging
logger = logging.getLogger(__name__)


class ArrayStorage(object):
    def __init__(self, total_samples, variates, thin=1):
        self.__array__ = np.empty((int(total_samples / thin), variates))
        self.__current_index__ = 0
        self.__real_index__ = 0
        self.__thin__ = thin

    def add_sample(self, item):
        if self.__real_index__ % self.__thin__ == 0:
            if self.__current_index__ >= self.__array__.shape[0]:
                logger.warn('Trying to write to an index outside of the bounds of the array!')
            else:
                self.__array__[self.__current_index__, :] = item
                self.__current_index__ += 1

        self.__real_index__ += 1

    @property
    def array(self):
        return self.__array__
