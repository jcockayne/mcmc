import numpy as np
import logging
logger = logging.getLogger(__name__)


class ArrayStorage(object):
    def __init__(self, total_samples, variates, log_likelihoods=False, thin=1):
        thinned_samples = int(total_samples / thin)

        self.__array__ = np.empty((thinned_samples, variates))
        if log_likelihoods:
            self.__log_likelihoods__ = np.empty(thinned_samples)
        else:
            self.__log_likelihoods__ = None
        self.__current_index__ = 0
        self.__real_index__ = 0
        self.__thin__ = thin

    def add_sample(self, sample, sample_likelihood):
        if self.__real_index__ % self.__thin__ == 0:
            if self.__current_index__ >= self.__array__.shape[0]:
                logger.warn('Trying to write to an index outside of the bounds of the array!')
            else:
                self.__array__[self.__current_index__, :] = sample
                if self.__log_likelihoods__ is not None:
                    self.__log_likelihoods__[self.__current_index__] = sample_likelihood
                self.__current_index__ += 1

        self.__real_index__ += 1

    @property
    def array(self):
        return self.__array__
