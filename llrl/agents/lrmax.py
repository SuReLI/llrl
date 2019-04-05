import random
import copy
import numpy as np
from collections import defaultdict

from llrl.agents.rmax import RMax


class LRMax(RMax):
    """
    Lipschitz R-Max agent
    """

    def __init__(
            self,
            actions,
            gamma=0.9,
            count_threshold=1,
            epsilon=0.1,
            max_memory_size=None,
            prior=np.Inf,
            name="LRMax-e"
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
        RMax.__init__(self, actions, gamma, count_threshold, epsilon, name)

        # Lifelong Learning memories
        self.max_memory_size = max_memory_size
        self.U_memory = []
        self.R_memory = []
        self.T_memory = []

        self.prior = prior

        self.U_lip = []
        self.update_lipschitz_upper_bounds()

    def reset(self):
        """
        Reset the attributes to initial state (called between instances).
        Save the previous model.
        :return: None
        """
        # Save previously learned model
        if len(self.counter) > 0 and (self.max_memory_size is None or len(self.U_lip) < self.max_memory_size):
                self.update_memory()

        self.U, self.R, self.T, self.counter = self.empty_memory_structure()
        self.prev_s = None
        self.prev_a = None

        self.update_lipschitz_upper_bounds()

    def act(self, s, r):
        """
        Acting method called online during learning.
        :param s: int current state of the agent
        :param r: float received reward for the previous transition
        :return: return the greedy action wrt the current learned model.
        """
        self.update(self.prev_s, self.prev_a, r, s)

        a = self.greedy_action(s, self.min_upper_bound(s))

        self.prev_a = a
        self.prev_s = s

        return a

    def min_upper_bound(self, s):
        """
        Choose the minimum local upper-bound between the one provided by R-Max and
        those provided by each Lipschitz bound.
        :param s: input state for which the bound is derived
        :return: return the minimum upper-bound.
        """
        u_min = copy.deepcopy(self.U)
        for u_lip in self.U_lip:
            for a in self.actions:
                if u_lip[s][a] < u_min[s][a]:
                    u_min[s][a] = copy.deepcopy(u_lip[s][a])
        return u_min

    def update_memory(self):
        """
        Update the memory (called between each MDP change i.e. when the method reset is called).
        Store the rewards, transitions and upper-bounds for the known state-action pairs
        respectively in R_memory, T_memory and U_memory.
        All the data corresponding to partially known state-action pairs are discarded.
        Consequently, the saved state-action pairs only refer to known pairs.
        :return: None
        """
        new_r = defaultdict(lambda: defaultdict(list))
        new_t = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        new_u = defaultdict(lambda: defaultdict(lambda: self.r_max / (1. - self.gamma)))

        for s in self.R:
            for a in self.R[s]:
                if self.is_known(s, a):
                    new_r[s][a] = self.R[s][a]
                    new_t[s][a] = self.T[s][a]

        for s in new_r:
            for a in new_r[s]:
                new_u[s][a] = self.U[s][a]

        self.R_memory.append(new_r)
        self.T_memory.append(new_t)
        self.U_memory.append(new_u)

    def update_lipschitz_upper_bounds(self):
        """
        Update the Lipschitz upper-bound for all the previous tasks.
        Called at initialization and when a new state-action pair is known.
        :return: None
        """
        n_prev_mdps = len(self.U_memory)
        self.U_lip = []
        if n_prev_mdps > 0:
            for i in range(n_prev_mdps):
                self.U_lip.append(
                    self.compute_lipschitz_upper_bound(self.U_memory[i], self.R_memory[i], self.T_memory[i])
                )

    def compute_lipschitz_upper_bound(self, U_mem, R_mem, T_mem):
        # 1. Separate state-action pairs
        s_a_kk, s_a_ku, s_a_uk = self.separate_state_action_pairs(R_mem)

        # 2. Compute models distances upper-bounds
        d_dict = self.models_distances(U_mem, R_mem, T_mem, s_a_kk, s_a_ku, s_a_uk)

        # 3. Compute the Q-values gap with dynamic programming
        gap = self.q_values_gap(d_dict, s_a_kk, s_a_ku, s_a_uk)

        # 4. Deduce upper-bound from U_mem
        return self.lipschitz_upper_bound(U_mem, gap)

    def separate_state_action_pairs(self, R_mem):
        """
        Create 3 lists of state-action pairs corresponding to:
        - pairs known in the current MDP and the considered previous one;
        - known only in the current MDP;
        - known only in the previous MDP.
        :param R_mem: Reward memory of the previous MDP
        :return: the 3 lists as a tuple
        """
        # Different state-action pairs container:
        s_a_kk = []  # Known in both MDPs
        s_a_ku = []  # Known in current MDP - Unknown in previous MDP
        s_a_uk = []  # Unknown in current MDP - Known in previous MDP

        # Fill containers
        for s in self.R:
            for a in self.actions:
                if self.is_known(s, a):
                    if s in R_mem and a in R_mem[s]:  # (s, a) known for both MDPs
                        s_a_kk.append((s, a))
                    else:  # (s, a) only known in current MDP
                        s_a_ku.append((s, a))
        for s in R_mem:
            for a in R_mem[s]:
                if not self.is_known(s, a):  # (s, a) only known in previous MDP
                    s_a_uk.append((s, a))

        return s_a_kk, s_a_ku, s_a_uk

    def models_distances(self, U_mem, R_mem, T_mem, s_a_kk, s_a_ku, s_a_uk):
        # Initialize model's distances upper-bounds
        d_dict = defaultdict(lambda: defaultdict(lambda: self.r_max * (1. + self.gamma) / (1. - self.gamma)))

        # Compute model's distances upper-bounds for known-known (s, a)
        for s, a in s_a_kk:
            n_s_a = float(self.counter[s][a])
            r_s_a = sum(self.R[s][a]) / n_s_a
            r_s_a_mem = sum(R_mem[s][a]) / float(self.count_threshold)  # there should be count_threshold rewards

            assert self.count_threshold == len(R_mem[s][a])  # TODO remove after testing

            weighted_sum = 0.
            for s_p in self.T[s][a]:
                if s_p in T_mem[s][a]:
                    a_p = self.greedy_action(s_p, U_mem)
                    model_diff = abs(self.T[s][a][s_p] / n_s_a - T_mem[s][a][s_p] / float(self.count_threshold))
                    # TODO check that T_mem[s][a][s_p] == 0 if s_p not in T_mem[s][a]
                    weighted_sum += U_mem[s_p][a_p] * model_diff
            for s_p in T_mem[s][a]:
                if not s_p in self.T[s][a]:
                    a_p = self.greedy_action(s_p, U_mem)
                    model_diff = T_mem[s][a][s_p] / float(self.count_threshold)
                    weighted_sum += U_mem[s_p][a_p] * model_diff
            d_dict[s][a] = abs(r_s_a - r_s_a_mem) + self.gamma * weighted_sum
            # TODO check this

        # Compute model's distances upper-bounds for known-unknown (s, a)
        for s, a in s_a_ku:
            n_s_a = float(self.counter[s][a])
            r_s_a = sum(self.R[s][a]) / n_s_a

            weighted_sum = 0.
            for s_p in self.T[s][a]:
                a_p = self.greedy_action(s_p, U_mem)
                weighted_sum += U_mem[s_p][a_p] * self.T[s][a][s_p] / n_s_a

            d_dict[s][a] = \
                max(self.r_max - r_s_a, r_s_a) + \
                self.gamma * weighted_sum + \
                self.r_max * self.gamma / (1. - self.gamma)
            # TODO check this

        # Compute model's distances upper-bounds for unknown-known (s, a)
        for s, a in s_a_uk:
            r_s_a_mem = sum(R_mem[s][a]) / float(self.count_threshold)  # there should be count_threshold rewards

            weighted_sum = 0.
            for s_p in T_mem[s][a]:
                a_p = self.greedy_action(s_p, U_mem)
                weighted_sum += U_mem[s_p][a_p] * T_mem[s][a][s_p] / float(self.count_threshold)

            d_dict[s][a] = \
                max(self.r_max - r_s_a_mem, r_s_a_mem) + \
                self.gamma * weighted_sum + \
                self.r_max * self.gamma / (1. - self.gamma)

        '''
        for s in d_dict:  # TODO investigate the large d_dict[s][a] when everything is unknown in new model
            for a in d_dict[s]:
                print(s, a, d_dict[s][a])
        '''

        return d_dict

    def q_values_gap(self, d_dict, s_a_kk, s_a_ku, s_a_uk):
        gap_max = self.r_max * (1. + self.gamma) / ((1. - self.gamma)**2)
        gap = defaultdict(lambda: defaultdict(lambda: gap_max))

        for s, a in s_a_uk:  # Unknown (s, a) in current MDP
            gap[s][a] = d_dict[s][a] + self.gamma * gap_max

        for i in range(self.vi_n_iter):
            for s, a in s_a_kk + s_a_ku:  # Known (s, a) in current MDP
                weighted_next_gap = 0.
                for s_p in self.T[s][a]:
                    a_p = self.greedy_action(s_p, gap)
                    weighted_next_gap += gap[s_p][a_p] * self.T[s][a][s_p] / float(self.counter[s][a])

                gap[s][a] = d_dict[s][a] + self.gamma * weighted_next_gap

        return gap

    def lipschitz_upper_bound(self, U_mem, gap):
        U = defaultdict(lambda: defaultdict(lambda: 2. * self.r_max / ((1. - self.gamma)**2)))
        for s in gap:
            for a in gap[s]:
                U[s][a] = U_mem[s][a] + gap[s][a]
        return U
