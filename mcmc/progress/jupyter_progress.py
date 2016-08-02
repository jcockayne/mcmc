import time
import ipywidgets
from IPython.display import display
import numpy as np
from ring_buffer import RingBuffer
from time_utils import pretty_time_delta
import base

update_frequency_seconds = 0.1
max_accept_lag = 1000

class JupyterProgress(base.ProgressBase):
    def __init__(self, recent_acceptance_lag=None, verbosity=1):
        self.start_time = None
        self.last_update_time = None
        self.verbosity = verbosity
        self.recent_acceptance_lag = recent_acceptance_lag
        self.last_update_iteration = 0
        self.n_iter = None
        self.__total_errors__ = 0

        self.__iter_time_buffer__ = RingBuffer(100)

        self.__text_field__ = ipywidgets.HTML()
        self.__error_field__ = None
        self.__error_label__ = None

    def initialise(self, n_iter):
        self.start_time = self.last_update_time = time.time()
        self.n_iter = n_iter

        if self.recent_acceptance_lag is None:
            self.recent_acceptance_lag = min(int(n_iter * 1. / 100), max_accept_lag)

        self.__text_field__.value = '<i>Waiting for {} iterations to have passed...</i>'.format(self.recent_acceptance_lag)
        #display(self.__progress_field__)
        display(self.__text_field__)

    def report_error(self, iter, error):
        if self.__error_field__ is None:
            self.__initialise_errors__()
        if self.verbosity > 0:
            self.__error_field__.value += 'Iteration {}: {}\n'.format(iter, error)
        self.__total_errors__ += 1

    def update(self, iteration, acceptances):
        if self.n_iter is None:
            raise Exception('First pass in the number of iterations!')

        recent_acceptance_lag = self.recent_acceptance_lag

        now = time.time()
        toc = now - self.last_update_time
        do_update = toc > update_frequency_seconds and iteration > self.last_update_iteration

        if do_update or iteration == self.n_iter:

            delta_accept = acceptances[-recent_acceptance_lag:].mean()*100 if iteration > recent_acceptance_lag else np.nan
            tot_accept = acceptances.mean()*100

            new_iterations = iteration - self.last_update_iteration
            time_per_iter = toc * 1./new_iterations
            self.__iter_time_buffer__.append(time_per_iter)

            # exclude outliers
            all_iter_times = np.array(self.__iter_time_buffer__.data)
            if len(all_iter_times) > 1:
                all_iter_times = all_iter_times[abs(all_iter_times - np.mean(all_iter_times)) < 2 * np.std(all_iter_times)]
                time_per_iter = all_iter_times.mean()

            eta = (self.n_iter - iteration) * time_per_iter

            html = self.__get_text_field__(iteration, self.n_iter, delta_accept, tot_accept, time_per_iter, eta)
            self.__text_field__.value = html

            self.last_update_time = now
            self.last_update_iteration = iteration

    def __initialise_errors__(self):
        if self.verbosity > 0:
            self.__error_label__ = ipywidgets.HTML(value="<span style='color: red'>Errors:</span>")
            self.__error_field__ = ipywidgets.Textarea(disabled=True)
            display(self.__error_label__)
            display(self.__error_field__)

    def __get_text_field__(self, iteration, n_iter, delta_accept, total_accept, time_per_iter, eta):
        template = """
                <div class="progress">
                  <div class="progress-bar" role="progressbar" aria-valuenow="{}"
                  aria-valuemin="0" aria-valuemax="{}" style="width:{:.2f}%">
                    <span class="sr-only">{:.2f}% Complete</span>
                  </div>
                </div>
                <table class="table">
                        <tr>
                                <td>Current iteration</td>
                                <td>{}</td>
                        </tr>
                        <tr>
                                <td>Accept rate (last {})</td>
                                <td>{:.2f}%</td>
                        </tr>
                        <tr>
                                <td>Accept rate (overall)</td>
                                <td>{:.2f}%</td>
                        </tr>
                        <tr>
                                <td>T/iter</td>
                                <td>{:.4f} seconds</td>
                        </tr>
                        <tr>
                                <td>ETA</td>
                                <td>{}</td>
                        </tr>
                        <tr>
                                <td>Errors</td>
                                <td>{}</td>
                        </tr>
                </table>
        """
        return template.format(iteration,
                               n_iter,
                               iteration*100./n_iter,
                               iteration*100./n_iter,
                               iteration,
                               self.recent_acceptance_lag,
                               delta_accept,
                               total_accept,
                               time_per_iter,
                               pretty_time_delta(eta),
                               self.__total_errors__)