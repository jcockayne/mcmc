import numpy as np
import progress


def hmc(iterations, log_pi, epsilon, L, q_0, grad_log_pi=None, progress_object=None):
    progress_object = progress.factory(progress_object)
    if grad_log_pi is None:
        try:
            import autograd
            grad_log_pi = autograd.grad(log_pi)
        except Exception as ex:
            raise Exception('Gradient of log target not passed and unable to use autograd. {}'.format(ex))

    res = np.empty((iterations, q_0.shape[0]))
    acceptances = np.empty(iterations, dtype=np.bool)

    U = lambda x: -log_pi(x)
    grad_U = lambda x: -grad_log_pi(x)

    cur_q = q = q_0

    progress_object.initialise(iterations)

    for i in xrange(iterations):
        p = np.random.normal(size=q.shape[0])
        cur_p = p
        # half step
        try:
            p -= epsilon * grad_U(q) / 2.

            # alternate full steps for position and momentum
            for j in xrange(L):
                q += epsilon * p
                if i != L:
                    p -= epsilon * grad_U(q)

            # now half-step
            p -= epsilon * grad_U(q) / 2.

            # negate momentum to make the proposal symmetric
            p = -p

            # evaluate potential and kinetic energies at start and end of trajectory
            cur_U = U(cur_q)
            cur_K = np.sum(cur_p**2) / 2.
            new_U = U(q)
            new_K = np.sum(p**2) / 2.
            alpha = min(1, np.exp(-new_U + cur_U - new_K + cur_K))

            accept = np.random.uniform() < alpha
        except Exception as ex:
            progress_object.report_error(i, ex)
            accept = False

        # accept / reject step
        if accept:
            res[i, :] = q
            cur_q = q
        else:
            res[i, :] = cur_q
        acceptances[i] = accept

        progress_object.update(i, acceptances[:(i+1)])

    progress_object.update(iterations, acceptances)
    return res