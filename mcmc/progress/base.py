import abc


class ProgressBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def initialise(self, n_iter):
        pass

    @abc.abstractmethod
    def report_error(self, iter, error):
        pass

    @abc.abstractmethod
    def update(self, iteration, acceptances):
        pass
