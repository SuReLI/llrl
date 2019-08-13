import numpy as np


def experiment():
    err = []
    n_states = 3
    n_samples = int(1e5)
    n_known = 100

    # True distribution
    p = np.random.random((n_states,))
    p /= np.sum(p)
    # p = np.array([0.1, 0.5, 0.4])

    # Learn n_samples models
    for i in range(n_samples):

        # Generate n_known samples from the true model
        out = np.random.choice(n_states, size=(n_known,), p=p)

        # Learn the model
        p_i = np.zeros((n_states,))
        for o in out:
            p_i[o] += 1
        p_i = p_i / float(n_known)

        # For each learned model, compute the error
        e_i = np.sum(np.abs(p - p_i))
        err.append(e_i)

    # Print the results
    print('Max error in L1-norm  :', max(err))
    print('Mean error in L1-norm :', np.mean(err))


if __name__ == '__main__':
    np.random.seed(1993)
    experiment()
