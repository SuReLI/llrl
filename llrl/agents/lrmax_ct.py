import random
import numpy as np
import copy
from collections import defaultdict

from llrl.agents.lrmax import LRMax


class LRMaxCT(LRMax):
    """
    Lipschitz R-Max agent for constant transitions setting.

    Assume shared transition function across MDPs in Lifelong RL setting.
    """

    def __init__(
            self,
            actions,
            gamma=.9,
            count_threshold=1,
            epsilon=.1,
            max_memory_size=None,
            prior=1.,
            name="LRMax-CT"
    ):
        """
        :param actions: action space of the environment
        :param gamma: (float) discount factor
        :param count_threshold: (int) count after which a state-action pair is considered known
        :param epsilon: (float) precision of value iteration algorithm
        :param max_memory_size: (int) maximum number of saved models (infinity if None)
        :param prior: (float) prior knowledge of maximum model's distance
        :param name: (str)
        """
        LRMax.__init__(
            self,
            actions=actions,
            gamma=gamma,
            count_threshold=count_threshold,
            epsilon=epsilon,
            max_memory_size=max_memory_size,
            prior=prior,
            name=name
        )

    def compute_lipschitz_upper_bound(self, U_mem, R_mem, T_mem):
        ''' Note: different from LRMax '''
        # 1. Separate state-action pairs
        s_a_kk, s_a_ku, s_a_uk = self.separate_state_action_pairs(R_mem)

        # 2. Compute models distances upper-bounds
        d_dict = self.models_distances(R_mem, s_a_kk, s_a_ku, s_a_uk)

        # 3. Compute the Q-values gap with dynamic programming
        gap = self.q_values_gap(d_dict, T_mem, s_a_kk, s_a_ku, s_a_uk)

        # 4. Deduce upper-bound from U_mem
        return self.lipschitz_upper_bound(U_mem, gap)

    def models_distances(self, R_mem, s_a_kk, s_a_ku, s_a_uk):
        ''' Note: different from LRMax '''
        # Initialize model's distances upper-bounds
        d_dict = defaultdict(lambda: defaultdict(lambda: self.prior))

        # Compute model's distances upper-bounds for known-known (s, a)
        for s, a in s_a_kk:
            n_s_a = float(self.counter[s][a])
            r_s_a = sum(self.R[s][a]) / n_s_a
            r_s_a_mem = sum(R_mem[s][a]) / float(self.count_threshold)
            d_dict[s][a] = abs(r_s_a - r_s_a_mem)
            assert self.count_threshold == len(R_mem[s][a])  # TODO remove after testing

        # Compute model's distances upper-bounds for known-unknown (s, a)
        for s, a in s_a_ku:
            r_s_a = sum(self.R[s][a]) / float(self.count_threshold)
            d_dict[s][a] = min(self.prior, max(self.r_max - r_s_a, r_s_a))
            assert self.count_threshold == self.counter[s][a]  # TODO remove after testing

        # Compute model's distances upper-bounds for unknown-known (s, a)
        for s, a in s_a_uk:
            r_s_a_mem = sum(R_mem[s][a]) / float(self.count_threshold)
            d_dict[s][a] = min(self.prior, max(self.r_max - r_s_a_mem, r_s_a_mem))
            assert self.count_threshold == len(R_mem[s][a])  # TODO remove after testing

        return d_dict

    def q_values_gap(self, d_dict, T_mem, s_a_kk, s_a_ku, s_a_uk):
        ''' Note: different from LRMax '''
        gap_max = self.prior / (1. - self.gamma)
        gap = defaultdict(lambda: defaultdict(lambda: gap_max))

        for s, a in s_a_uk:  # Unknown (s, a) in current MDP
            gap[s][a] = d_dict[s][a] + self.gamma * gap_max

        for i in range(self.vi_n_iter):
            for s, a in s_a_uk:  # Unknown (s, a) in current MDP
                weighted_next_gap = 0.
                for s_p in T_mem[s][a]:
                    a_p = self.greedy_action(s_p, gap)
                    weighted_next_gap += gap[s_p][a_p] * T_mem[s][a][s_p] / float(self.count_threshold)

                gap[s][a] = d_dict[s][a] + self.gamma * weighted_next_gap

            for s, a in s_a_kk + s_a_ku:  # Known (s, a) in current MDP
                weighted_next_gap = 0.
                for s_p in self.T[s][a]:
                    a_p = self.greedy_action(s_p, gap)
                    weighted_next_gap += gap[s_p][a_p] * self.T[s][a][s_p] / float(self.count_threshold)
                    assert self.count_threshold == self.counter[s][a]  # TODO remove after testing

                gap[s][a] = d_dict[s][a] + self.gamma * weighted_next_gap

        return gap

    def lipschitz_upper_bound(self, U_mem, gap):
        ''' Note: different from LRMax '''
        U = defaultdict(lambda: defaultdict(lambda: (self.prior + self.r_max) / (1. - self.gamma)))
        for s in gap:
            for a in gap[s]:
                U[s][a] = U_mem[s][a] + gap[s][a]
        return U
