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
        self.cr = 0.1  # mixing parameter for reward term in bi-simulation metric
        self.ct = 1.0 - self.cr  # mixing parameter for transition term in bi-simulation metric

    def best_match_mdp_distance(self, m1, m2, d=None, threshold=0.1):
        """
        Compute the best-match distance between two MDPs i.e. find the best matching states using the input state
        distance matrix and return the maximum distance between matched states.

        Matching states matrix: in row, the states in MDP m1, in columns, the states in MDP m2.

        :param m1: 1st MDP (environment)
        :param m2: 2nd MDP (environment)
        :param d: state distance matrix, compute the bi-simulation distance matrix if not provided
        :param threshold: threshold for the bi-simulation distance matrix computation
        :return: return the tuple (distance between the MDPs, matching states matrix)
        """
        assert m1.nS == m2.nS, 'Error: environments have different number of states: m1.nS={}, m2.nS={}'.format(m1.nS,
                                                                                                                m2.nS)
        ns = m1.nS
        if d is None:
            d = self.bi_simulation_distance(m1, m2, threshold)

        pb = pulp.LpProblem('best-match', pulp.LpMinimize)
        l = np.empty(d.shape, dtype=object)
        for i in range(ns):
            for j in range(ns):
                l[i, j] = pulp.LpVariable('l_{}_{}'.format(i, j), cat=pulp.LpBinary)
        pb += np.sum(l * d)
        for i in range(ns):  # Sum to 1
            pb += np.sum(l[i, :]) == 1
            pb += np.sum(l[:, i]) == 1
        assert pulp.LpStatus[pb.solve()] == 'Optimal'
        matching_matrix = np.array([[l[i, j].value() for j in range(ns)] for i in range(ns)])
        distance = 0.0
        for i in range(ns):
            for j in range(ns):
                if matching_matrix[i, j] == 1:
                    if d[i, j] > distance:
                        distance = d[i, j]
        return distance, matching_matrix

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
        assert m1.nS == m2.nS, "Error: environments have different number of states: m1.nS={}, m2.nS={}".format(m1.nS,
                                                                                                                m2.nS)
        if d is None:
            d = self.bi_simulation_distance(m1, m2, threshold)
        ns = m1.nS
        uniform_distribution = (1.0 / float(ns)) * np.ones(shape=ns, dtype=float)
        distance, matching_matrix = distribution.wass_primal(uniform_distribution, uniform_distribution, d)
        matching_matrix = np.reshape(matching_matrix, newshape=(ns, ns))
        return distance, matching_matrix

    def bi_simulation_distance(self, m1, m2, threshold=0.1):
        assert m1.nS == m2.nS, 'Error: environments have different number of states: m1.nS={}, m2.nS={}'.format(m1.nS,
                                                                                                                m2.nS)
        assert m1.nA == m2.nA, 'Error: environments have different number of actions: m1.nA={}, m2.nA={}'.format(m1.nA,
                                                                                                                 m2.nA)
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
                        delta_t, _ = distribution.wass_primal(di, dj, d)
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
