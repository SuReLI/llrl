"""
Environments handler

Required features:
env.nS
env.expected_reward(s, a)
env.transition_probability_distribution(s, a)
"""

import numpy as np
import llrl.utils.distribution as distribution


class Handler(object):
    def __init__(self):
        self.cr = 0.5  # mixing parameter for reward term in bi-simulation metric
        self.ct = 0.5  # mixing parameter for transition term in bi-simulation metric

    def mdp_distance(self, m1, m2, d=None, threshold=0.1):
        assert m1.nS == m2.nS, 'Error: environments have different number of states: m1.nS={}, m2.nS={}'.format(m1.nS, m2.nS)
        if d is None:
            d = self.bi_simulation_distance(m1, m2, threshold)
        ns = m1.nS
        # uniform_distribution = (1.0 / float(ns)) * np.ones(shape=ns, dtype=float)
        uniform_distribution = np.ones(shape=ns, dtype=float)
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
