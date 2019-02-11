"""
Helpful functions when dealing with distributions.
"""

import numpy as np
import llrl.utils.utils as utils
from itertools import combinations
from scipy.optimize import linprog


def marginal_matrices(n):
    A = np.zeros(shape=(n, n ** 2))
    B = np.zeros(shape=(n, n ** 2))
    for i in range(n):
        A[i][i * n:(i + 1) * n] = 1
        for j in range(n):
            B[i][j * n + i] = 1
    return A, B


def wass_primal(u, v, d):
    """
    Compute the 1-Wasserstein distance between u (shape=n) and v (shape=n) given the distances matrix d (shape=(n,n)).
    Use the primal formulation.
    """
    n = d.shape[0]
    obj = np.reshape(d, newshape=(n * n))
    A, B = marginal_matrices(n)
    Ae = np.concatenate((A, B), axis=0)
    be = np.concatenate((u, v))
    res = linprog(obj, A_eq=Ae, b_eq=be)
    return res.fun


def wass_dual(u, v, d):
    """
    Compute the 1-Wasserstein distance between u (shape=n) and v (shape=n) given the distances matrix d (shape=(n,n)).
    Use the dual formulation.
    """
    n = d.shape[0]
    comb = np.array(list(combinations(range(n), 2)))
    obj = u - v
    Au = np.zeros(shape=(n * (n - 1), n))
    bu = np.zeros(shape=(n * (n - 1)))
    for i in range(len(comb)):
        Au[2 * i][comb[i][0]] = +1.0
        Au[2 * i][comb[i][1]] = -1.0
        Au[2 * i + 1][comb[i][0]] = -1.0
        Au[2 * i + 1][comb[i][1]] = +1.0
        bu[2 * i] = d[comb[i][0]][comb[i][1]]
        bu[2 * i + 1] = d[comb[i][0]][comb[i][1]]
    res = linprog(obj, A_ub=Au, b_ub=bu)
    return -res.fun


def random_tabular(size):
    """
    Generate a 1D numpy array whose coefficients sum to 1
    """
    w = np.random.random(size)
    return w / np.sum(w)


def random_constrained(u, d, maxdist):
    """
    Randomly generate a new distribution st the Wasserstein distance between the input
    distribution u and the generated distribution is smaller than the input maxdist.
    The distance is computed w.r.t. the distances matrix d.
    Notice that the generated distribution has the same support as the input distribution.
    """
    max_n_trial = int(1e4)  # Maximum number of trials
    val = np.asarray(range(len(u)))
    v = random_tabular(val.size)
    for i in range(max_n_trial):
        if wass_dual(u, v, d) <= maxdist:
            return v
        else:
            v = random_tabular(val.size)
    print('Failed to generate constrained distribution after {} trials'.format(max_n_trial))
    exit()


def clean_distribution(w):
    """
    Set weights to 0 or 1 if close (correct numerical mistakes).
    """
    for i in range(len(w)):
        assert w[i] >= 0.0, 'Error: negative weight: w[{}]={}'.format(i, w[i])
        assert w[i] <= 1.0, 'Error: weight bigger than 1: w[{}]={}'.format(i, w[i])
        if utils.close(w[i], 0.0):
            w[i] = 0.0
        elif utils.close(w[i], 1.0):
            w[i] = 1.0
    return w


def worstcase_distribution_direct_method(v, w0, c, d):
    n = len(v)
    if utils.close(c, 0.0) or utils.closevec(v, v[0] * np.ones(n)):
        return w0
    w_worst = np.zeros(n)
    w_worst[np.argmin(v)] = 1.0
    if wass_dual(w_worst, w0, d) <= c:
        return w_worst
    lbd = c / wass_dual(w0, w_worst, d)
    w = (1.0 - lbd) * w0 + lbd * w_worst
    return clean_distribution(w)
