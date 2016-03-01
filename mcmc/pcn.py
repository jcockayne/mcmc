from __future__ import print_function
import numpy as np
import time, sys
import progress
from utilities import as_single_number
import storage as st

class PCNProposal(object):
    def __init__(self, beta, covariance_matrix):
        self.beta = beta
        (u, s, v) = np.linalg.svd(covariance_matrix)
        self.__dot_with_xi = np.sqrt(s)[:, None] * v

    def __call__(self, current):
        xi = np.dot(np.random.normal(size=len(current)), self.__dot_with_xi)
        xi = xi.reshape(current.shape)
        new = np.sqrt(1-self.beta**2)*current + self.beta*xi
        return new


def proposal(beta, covariance_matrix):
    return PCNProposal(beta, covariance_matrix)


def adapt_function(adapt_frequency, min_accept, max_accept, factor, verbosity=1):
    def __do_adaptation(cur_proposal, samples, acceptances):
        new_accept = acceptances[-adapt_frequency:].mean()
        if verbosity > 1:
            print('Current acceptance ratio: {:.4f}'.format(new_accept))
        if new_accept < min_accept:
            cur_proposal.beta /= factor
        elif new_accept > max_accept:
            cur_proposal.beta *= factor
        if new_accept < min_accept or new_accept > max_accept and verbosity > 0:
            print('Updated beta to {}'.format(cur_proposal.beta))

        return cur_proposal
    return __do_adaptation


def pCN(iterations, propose, phi, kappa_0, adapt_frequency=None, adapt_function=None, progress_object=None, storage=None):
    if progress_object is None:
        progress_object = progress.get_default_progress()

    if adapt_frequency is not None and adapt_function is None:
        raise Exception('Adapt frequency supplied but no adapt function specified.')

    # create an empty numpy if the array is not supplied
    return_array = storage is None
    if storage is None:
        storage = st.DiskBackedStorage((iterations, kappa_0.shape[0]))

    acceptances = np.empty(iterations, dtype=np.bool)

    cur_kappa = kappa_0
    cur_phi = as_single_number(phi(cur_kappa))

    progress_object.initialise(iterations)

    for i in xrange(iterations):
        if adapt_frequency is not None and i > 0 and i % adapt_frequency == 0:
            propose = adapt_function(propose, storage, acceptances)

        new_kappa = propose(cur_kappa)
        new_phi = as_single_number(phi(new_kappa))

        if np.isnan(new_phi):
            progress_object.report_error(i, 'About to reject proposal because new value of phi is NaN.'.format(i))
            accept = False
        else:
            alpha = min(1, np.exp(cur_phi-new_phi))
            accept = alpha > np.random.uniform()
        if accept:
            cur_kappa = new_kappa
            cur_phi = new_phi

        storage.set_item(i, cur_kappa.ravel())
        acceptances[i] = accept

        progress_object.update(i, acceptances[:i])
    progress_object.update(iterations, acceptances[:i])
    if return_array:
        return storage.array
    return storage
