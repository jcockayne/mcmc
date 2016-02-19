import numpy as np
import time, sys

def proposal(beta, covariance_matrix):
    (u, s, v) = np.linalg.svd(covariance_matrix)
    dot_with_xi = np.sqrt(s)[:, None] * v

    def __do_propose(current):
        xi = np.dot(np.random.normal(size=covariance_matrix.shape[0]), dot_with_xi)
        xi = xi.reshape(current.shape)
        new = np.sqrt(1-beta**2)*current + beta*xi
        return new
    return __do_propose


def rwm(iterations, propose, log_likelihood, log_prior, update_frequency=None, verbosity=1):
    phi = lambda x: -log_likelihood(x) - log_prior(x)
    return pCN(iterations, propose, phi, update_frequency, verbosity)


def pCN(iterations, propose, phi, kappa_0, update_frequency=None, verbosity=1):
    if update_frequency is None:
        update_frequency = int(iterations / 100)

    # now the MCMC
    kappas = np.empty((iterations, kappa_0.shape[0]))

    acceptances = np.empty(iterations, dtype=np.bool)

    cur_kappa = kappa_0
    cur_phi = phi(cur_kappa)
    if type(cur_phi) is np.ndarray:
        cur_phi = cur_phi.item()
    tic = time.time()
    for i in xrange(iterations):
        if verbosity == 1 and update_frequency > 0 and i % update_frequency == 0 and i > 0:
            toc = time.time() - tic
            delta_accept = acceptances[(i-update_frequency):i].mean()*100
            tot_accept = acceptances[:i].mean()*100
            print 'Iter {}: Accept ({:.0f}% {:.0f}%) T/Iter {:.4f}'.format(i, delta_accept, tot_accept, toc / update_frequency)
            sys.stdout.flush()
            tic = time.time()

        new_kappa = propose(cur_kappa)
        new_phi = phi(new_kappa)
        if type(new_phi) is np.ndarray:
            new_phi = new_phi.item()
        alpha = min(1, np.exp(cur_phi-new_phi))
        accept = alpha > np.random.uniform()
        if accept:
            cur_kappa = new_kappa
            cur_phi = new_phi

        if verbosity == 2:
            toc = time.time() - tic
            delta_accept = acceptances[(i-update_frequency):i].mean()*100 if i > update_frequency else np.nan
            tot_accept = acceptances[:i].mean()*100
            print 'Iter {}:  Pot {} -> {} Accept ({:.4f}, {:d}, {:.0f}%, {:.0f}%) T/Iter {:.4f}'.format(i, cur_phi, new_phi, alpha, accept, delta_accept, tot_accept, toc)
            sys.stdout.flush()
            tic = time.time()
        kappas[i, :] = cur_kappa.ravel()
        acceptances[i] = accept
    return kappas
