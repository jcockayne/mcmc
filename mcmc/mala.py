import numpy as np
from utilities import as_single_number
import progress

# TODO: This is a hacky cut/paste of RWM
# it seems like I should be able to do something a lot smarter than this...


def mala_proposal(theta, grad_log_likelihood, sigma):
    return theta + sigma**2 / 2. * grad_log_likelihood + sigma * np.random.normal(size=theta.shape)


def mala(iterations, sigma, log_likelihood, log_prior, init_theta, grad_log_likelihood=None, progress_object=None):
    if progress_object is None:
        progress_object = progress.get_default_progress()
    if grad_log_likelihood is None:
        # TODO: might want to return gradient from log_likelihood function directly...
        try:
            import autograd
        except:
            raise Exception('Autograd not available and gradient function not supplied; cannot use MALA.')
        grad_log_likelihood = autograd.grad(log_likelihood)

    if type(init_theta) is np.ndarray:
        theta_shape = init_theta.shape[0]
    else:
        theta_shape = 1
    samples = np.empty((iterations, theta_shape))
    acceptances = np.empty(iterations, dtype=np.bool)

    cur_theta = init_theta
    cur_log_likelihood = as_single_number(log_likelihood(cur_theta))
    cur_grad_log_likelihood = grad_log_likelihood(cur_theta)
    cur_log_prior = as_single_number(log_prior(cur_theta))

    progress_object.initialise(iterations)

    for i in xrange(iterations):
        new_theta = mala_proposal(cur_theta, cur_grad_log_likelihood, sigma)
        new_log_prior = as_single_number(log_prior(new_theta))

        if np.isinf(new_log_prior) and new_log_prior < 0:
            accept = False
        else:
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

        if accept:
            cur_theta = new_theta
            cur_log_likelihood = new_log_likelihood
            cur_grad_log_likelihood = grad_log_likelihood(cur_theta)
        samples[i, :] = cur_theta
        acceptances[i] = accept

        progress_object.update(i, acceptances[:(i+1)])
    progress_object.update(iterations, acceptances)
    return samples