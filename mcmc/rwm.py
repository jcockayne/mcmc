import numpy as np
from utilities import as_single_number
import progress


def mala_proposal(grad_pi, sigma):
    def __proposal(theta):
        return theta + sigma * np.random.normal(size=theta.shape) + sigma**2 / 2. * grad_pi(theta)
    return __proposal


class AdaptiveProposal(object):
    def __init__(self, initial_variance, adapt_rate=100, adapt_factor=2.0, adapt_limits=(0.2,0.5)):
        self.adapt_rate = adapt_rate
        self.adapt_factor = adapt_factor
        self.adapt_limits = adapt_limits
        self.variance = initial_variance
        self.accepts = 0

    def __call__(self, param):
        return np.random.normal(param, self.variance)

    def adapt(self, iteration, accepted):
        self.accepts += accepted

        if iteration > 0 and iteration % self.adapt_rate == 0:
            accept_rate = self.accepts * 1. / self.adapt_rate
            if self.adapt_limits[0] > accept_rate:
                self.variance /= self.adapt_factor
            if self.adapt_limits[1] < accept_rate:
                self.variance *= self.adapt_factor
            self.accepts = 0


def rwm(iterations, propose, log_likelihood, log_prior, init_theta, progress_object=None):
    progress_object = progress.factory(progress_object)

    adapt = hasattr(propose, 'adapt')

    if type(init_theta) is np.ndarray:
        theta_shape = init_theta.shape[0]
    else:
        theta_shape = 1
    samples = np.empty((iterations, theta_shape))
    acceptances = np.empty(iterations, dtype=np.bool)

    cur_theta = init_theta
    cur_log_likelihood = as_single_number(log_likelihood(cur_theta))
    cur_log_prior = as_single_number(log_prior(cur_theta))

    progress_object.initialise(iterations)

    for i in xrange(iterations):
        new_theta = propose(cur_theta)
        new_log_prior = as_single_number(log_prior(new_theta))

        if np.isinf(new_log_prior) and new_log_prior < 0:
            accept = False
        else:
            try:
                new_log_likelihood = log_likelihood(new_theta)
                new_log_likelihood = as_single_number(new_log_likelihood)

                if np.isnan(new_log_likelihood):
                    progress_object.report_error(i, 'Proposal is about to be rejected because likelihood is NaN')
                    accept = False
                elif np.isnan(new_log_prior):
                    progress_object.report_error(i, 'Proposal is about to be rejected because prior is NaN')
                    accept = False
                else:
                    alpha = min(1, np.exp(new_log_likelihood + new_log_prior - cur_log_likelihood - cur_log_prior))
                    accept = np.random.uniform() < alpha
            except Exception as ex:
                accept = False
                progress_object.report_error(i, ex)

        if accept:
            cur_theta = new_theta
            cur_log_likelihood = new_log_likelihood
            cur_log_prior = new_log_prior
        samples[i, :] = cur_theta
        acceptances[i] = accept

        if adapt:
            propose.adapt(i, accept)

        progress_object.update(i, acceptances[:(i+1)])
    progress_object.update(iterations, acceptances)
    return samples, acceptances
