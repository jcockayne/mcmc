def is_null_progress(progress_object):
    return progress_object.lower() == 'none'

class NullProgress(object):
    def initialise(self, n_iter):
        pass

    def report_error(self, iter, error):
        pass

    def update(self, iteration, acceptances):
        pass