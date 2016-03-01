import time
import sys


class PrintProgress(object):
    def __init__(self, update_frequency=None, verbosity=1):
        self.start_time = None
        self.last_update_time = None
        self.verbosity = verbosity
        self.update_frequency = update_frequency

    def initialise(self, n_iter):
        self.start_time = self.last_update_time = time.time()
        if self.update_frequency is None:
            self.update_frequency = int(n_iter / 100)

    def report_error(self, iter, error):
        print('Iter {}: {}'.format(iter, error))
        sys.stdout.flush()

    def update(self, iteration, acceptances):
        update_frequency = self.update_frequency

        now = time.time()
        toc = time.time() - now
        self.last_update_time = now

        delta_accept = acceptances[-update_frequency:].mean()*100
        tot_accept = acceptances.mean()*100

        if self.verbosity == 1 and iteration % self.update_frequency == 0 and iteration > 0:
            print('Iter {}: Accept ({:.0f}% {:.0f}%) T/Iter {:.4f}'.format(iteration, delta_accept, tot_accept, toc / update_frequency))
            sys.stdout.flush()