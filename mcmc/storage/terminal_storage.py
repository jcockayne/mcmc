class TerminalStorage(object):
    def __init__(self):
        self.__sample__ = None
        self.__log_likelihood__ = None

    def add_sample(self, sample, sample_likelihood):
        self.__sample__ = sample
        self.__log_likelihood__ = sample_likelihood

    @property
    def sample(self):
        return self.__sample__
    @property
    def log_likelihood(self):
        return self.__log_likelihood__
