import numpy as np
from utilities import as_single_number
import progress


class GibbsProposal(object):
    def __init__(self, proposal, indices):
        self.proposal = proposal
        self.indices = indices

    def __call__(self, x):
        return self.proposal(x)


def gibbs(iterations, proposals, log_likelihood, log_prior, init_theta, progress_object=None):
    progress_object = progress.factory(progress_object)

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

    propose_ix = 0
    proposal_indices = np.empty(iterations, dtype=np.int)
    for i in xrange(iterations):
        proposal = proposals[propose_ix]

        new_theta = cur_theta.copy()
        new_theta[proposal.indices] = proposal(cur_theta[proposal.indices])

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
        proposal_indices[i] = propose_ix
        propose_ix += 1
        if propose_ix == len(proposals):
            propose_ix = 0

        progress_object.update(i, acceptances[:(i+1)])

    progress_object.update(iterations, acceptances)
    return samples, acceptances, proposal_indices
