from __future__ import print_function
import numpy as np

import progress
from utilities import as_single_number
import storage as st
import logging
logger = logging.getLogger(__name__)
import scipy

class PCNProposal(object):

    def __init__(self, beta, covariance_matrix, prior_mean=None):
        self.beta = beta
        #(u, s, v) = np.linalg.svd(covariance_matrix)
        #self.__dot_with_xi = np.sqrt(s)[:, None] * v
        self.__prior_mean__ = np.zeros(covariance_matrix.shape[0]) if prior_mean is None else prior_mean
        self.__dot_with_xi__ = np.real_if_close(scipy.linalg.sqrtm(covariance_matrix))

    def __call__(self, current):
        xi = np.dot(self.__dot_with_xi__, np.random.normal(size=current.shape))
        new = self.__prior_mean__ + np.sqrt(1-self.beta**2)*(current - self.__prior_mean__) + self.beta*xi
        return new

class InfinityMalaProposal(object):
    def __init__(self, grad_phi, covariance_matrix, dt):
        #(u, s, v) = np.linalg.svd(covariance_matrix)
        #self.__sqrtm = np.sqrt(s)[:,None] * v
        self.__sqrtm = scipy.linalg.sqrtm(covariance_matrix)
        self.__covariance_matrix = covariance_matrix
        self.__dt = dt
        self.__grad_phi = grad_phi

        mat = covariance_matrix + 0.5*dt*np.eye(covariance_matrix.shape[0])
        term_a = np.linalg.inv(mat)
        term_b = covariance_matrix - 0.5*dt*np.eye(covariance_matrix.shape[0])
        self.__A_theta = term_a.dot(term_b)

        self.__B_theta = np.sqrt(2*dt)*term_a.dot(self.__sqrtm)

    def __call__(self, current):
        xi = np.dot(np.random.normal(size=len(current)), self.__sqrtm)
        gradient = self.__grad_phi(current)
        h = -np.sqrt(self.__dt/2.)*gradient
        return self.__A_theta.dot(current) + self.__B_theta.dot(xi + self.__sqrtm.dot(h))


def proposal(beta, covariance_matrix, prior_mean=None):
    return PCNProposal(beta, covariance_matrix, prior_mean)


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
    progress_object = progress.factory(progress_object)

    if adapt_frequency is not None and adapt_function is None:
        raise Exception('Adapt frequency supplied but no adapt function specified.')

    # create an empty numpy if the array is not supplied
    return_array = storage is None
    if storage is None:
        storage = st.ArrayStorage(iterations, kappa_0.shape[0])

    acceptances = np.empty(iterations, dtype=np.bool)

    cur_kappa = kappa_0
    cur_phi = as_single_number(phi(cur_kappa))

    progress_object.initialise(iterations)

    for i in xrange(iterations):
        if adapt_frequency is not None and i > 0 and i % adapt_frequency == 0:
            propose = adapt_function(propose, storage, acceptances)
        try:
            new_kappa = propose(cur_kappa)
            new_phi = as_single_number(phi(new_kappa))

            if np.isnan(new_phi):
                progress_object.report_error(i, 'About to reject proposal because new value of phi is NaN.'.format(i))
                accept = False
            else:
                alpha = min(1, np.exp(cur_phi-new_phi))
                accept = alpha > np.random.uniform()
        except Exception as ex:
            progress_object.report_error(i, ex)
            accept = False

        if accept:
            cur_kappa = new_kappa
            cur_phi = new_phi

        storage.add_sample(cur_kappa.ravel())
        acceptances[i] = accept

        progress_object.update(i, acceptances[:i])
    progress_object.update(iterations, acceptances)
    if return_array:
        return storage.array, acceptances
    return storage, acceptances
