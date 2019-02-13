"""
Environments handler

Required features:
env.nS
env.expected_reward(s, a)
env.transition_probability_distribution(s, a)
"""

import numpy as np
import pulp
import llrl.utils.distribution as distribution


class Handler(object):
    def __init__(self):
        self.cr = 0.5  # mixing parameter for reward term in bi-simulation metric
        self.ct = 0.5  # mixing parameter for transition term in bi-simulation metric

    def best_match_mdp_distance(self, m1, m2, d=None, threshold=0.1):
        """
        Compute the best-match distance between two MDPs i.e. find the best matching states using the input state
        distance matrix and return the maximum distance between matched states.

        :param m1: 1st MDP (environment)
        :param m2: 2nd MDP (environment)
        :param d: state distance matrix, compute the bi-simulation distance matrix if not provided
        :param threshold: threshold for the bi-simulation distance matrix computation
        :return: return the distance between the MDPs
        """
        assert m1.nS == m2.nS, 'Error: environments have different number of states: m1.nS={}, m2.nS={}'.format(m1.nS, m2.nS)
        # TODO put back
        # if d is None:
        #     d = self.bi_simulation_distance(m1, m2, threshold)

        # TODO remove test
        n = 2
        d = np.eye(n)
        print(d)
        exit()

    def wasserstein_mdp_distance(self, m1, m2, d=None, threshold=0.1):
        """
        Compute the Wasserstein distance between uniform distributions over the common state space using a specific
        state metric given by the input state distance matrix.

        :param m1: 1st MDP (environment)
        :param m2: 2nd MDP (environment)
        :param d: state distance matrix, compute the bi-simulation distance matrix if not provided
        :param threshold: threshold for the bi-simulation distance matrix computation
        :return: return the distance between the MDPs
        """
        assert m1.nS == m2.nS, 'Error: environments have different number of states: m1.nS={}, m2.nS={}'.format(m1.nS, m2.nS)
        if d is None:
            d = self.bi_simulation_distance(m1, m2, threshold)
        ns = m1.nS
        uniform_distribution = (1.0 / float(ns)) * np.ones(shape=ns, dtype=float)
        return distribution.wass_primal(uniform_distribution, uniform_distribution, d)

    def bi_simulation_distance(self, m1, m2, threshold=0.1):
        assert m1.nS == m2.nS, 'Error: environments have different number of states: m1.nS={}, m2.nS={}'.format(m1.nS, m2.nS)
        assert m1.nA == m2.nA, 'Error: environments have different number of actions: m1.nA={}, m2.nA={}'.format(m1.nA, m2.nA)
        ns, na = m1.nS, m1.nA
        d = np.zeros(shape=(ns, ns))
        tmp_d = np.array(d)
        gap = threshold + 1.0
        while gap > threshold:
            # Iterate
            for i in range(ns):
                for j in range(ns):
                    d_ija = 0.0
                    for a in range(na):
                        di = m1.transition_probability_distribution(i, a)
                        dj = m2.transition_probability_distribution(j, a)
                        # delta_t = distribution.wass_dual(di, dj, d)
                        delta_t = distribution.wass_primal(di, dj, d)
                        delta_r = abs(m1.expected_reward(i, a) - m2.expected_reward(j, a))
                        d_ija = max(d_ija, self.cr * delta_r + self.ct * delta_t)
                    tmp_d[i, j] = d_ija
            # Measure gap
            gap = 0.0
            for i in range(ns):
                for j in range(ns):
                    gap = max(gap, abs(d[i, j] - tmp_d[i, j]))
            # Update
            d = np.array(tmp_d)
        return d
