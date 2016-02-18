import numpy as np
import time

def proposal(beta, covariance_matrix):
    (u, s, v) = np.linalg.svd(covariance_matrix)
    dot_with_xi = np.sqrt(s)[:, None] * v

    def __do_propose(current):
        xi = np.dot(np.random.normal(size=covariance_matrix.shape[0]), dot_with_xi)
        xi = xi.reshape(current.shape)
        new = np.sqrt(1-beta**2)*current + beta*xi
        return new
    return __do_propose


def pCN(iterations, propose, phi, kappa_0, update_frequency=None, update_fun=None):
    if update_frequency is None:
        update_frequency = int(iterations / 100)

    # now the MCMC
    kappas = np.empty((iterations, kappa_0.shape[0]))

    acceptances = np.empty(iterations, dtype=np.bool)

    cur_kappa = kappa_0
    cur_phi = phi(cur_kappa)
    tic = time.time()
    for i in xrange(iterations):
        if update_frequency > 0 and i % update_frequency == 0 and i > 0:
            toc = time.time() - tic
            delta_accept = acceptances[(i-update_frequency):i].mean()
            tot_accept = acceptances[:i].mean()
            print 'Iter {}: Delta Accept {:.2f} Tot Accept {:.2f} T/Iter {:.4f}'.format(i, delta_accept, tot_accept, toc / update_frequency)
            tic = time.time()

        new_kappa = propose(cur_kappa)
        new_phi = phi(new_kappa)
        alpha = min(1, np.exp(cur_phi-new_phi))
        accept = alpha > np.random.uniform()
        if accept:
            cur_kappa = new_kappa
            cur_phi = new_phi

        kappas[i, :] = cur_kappa.ravel()
        acceptances[i] = accept
    return kappas
