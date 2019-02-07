"""
Dynamic programming algorithm

Required features:
env.nS
env.nA
env.expected_reward(s, a)
env.transition_probability_distribution(s, a)
"""

import numpy as np


def dynamic_programming(env, gamma=0.9, threshold=0.01, iter_max=1000, verbose=True):
    nS = env.nS
    nA = env.nA
    value_function = np.zeros(shape=nS, dtype=float)
    convergence, niter = False, 0
    for i in range(iter_max):
        tmp = (-1e99) * np.ones(shape=value_function.shape, dtype=float)
        for s in range(nS):
            for a in range(nA):
                v_a = env.expected_reward(s, a) + gamma * np.dot(env.transition_probability_distribution(s, a), value_function)
                tmp[s] = max(v_a, tmp[s])
        delta = 0.0
        for s in range(nS):
            delta = max(delta, abs(value_function[s] - tmp[s]))
        if verbose:
            print('Iteration', i+1, '/', iter_max, 'maximum difference =', delta)
        value_function = np.array(tmp)
        if delta < threshold:
            convergence, niter = True, i+1
            break
    if verbose:
        if convergence:
            print('Convergence after', niter, 'iterations.')
        else:
            print('Maximum number of iteration (', iter_max, ') reached before convergence.')
            print('Tolerance was', threshold)
    return value_function
