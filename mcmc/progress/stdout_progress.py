from __future__ import print_function
import time
import sys
import logging
logger = logging.getLogger(__name__)

class PrintProgress(object):
    def __init__(self, update_frequency=1000, verbosity=1):
        self.start_time = None
        self.last_update_time = None
        self.n_iter = None
        self.verbosity = verbosity
        self.update_frequency = update_frequency

    def initialise(self, n_iter):
        self.n_iter = n_iter
        self.start_time = self.last_update_time = time.time()
        self.last_update_time = time.time()

    def report_error(self, iter, error):
        print('Iter {}: {}'.format(iter, error))
        sys.stdout.flush()

    def update(self, iteration, acceptances):
        update_frequency = self.update_frequency

        toc = time.time() - self.last_update_time
        self.last_update_time = time.time()

        delta_accept = acceptances[-update_frequency:].mean()*100
        tot_accept = acceptances.mean()*100

        if self.verbosity == 1 and iteration % self.update_frequency == 0 and iteration > 0:
            remaining = toc / update_frequency * (self.n_iter - iteration)
            message = 'Iter {}: Accept ({:.0f}% {:.0f}%) T/Iter {:.4f} Remaining {}'.format(iteration, delta_accept, tot_accept, toc / update_frequency, remaining)
            print(message)
            sys.stdout.flush()
            logger.info(message)
