import random
import numpy as np
from collections import defaultdict

from simple_rl.agents.AgentClass import Agent


class LRMaxCT(Agent):
    """
    Lipschitz R-Max agent.

    Assumptions:
    - Shared transition function across MDPs in Lifelong RL setting;
    - Prior knowledge on maximal reward gap (attribute self.delta_r)
    """

    def __init__(self, actions, gamma=.9, count_threshold=1, epsilon=.1, delta_r=1., name="LRMaxCT"):
        ''' Note: different from LRMax '''
        name = name + "-e" + str(epsilon) + "-dr" + str(delta_r)
        Agent.__init__(self, name=name, actions=actions, gamma=gamma)
        self.r_max = 1.0
        self.count_threshold = count_threshold
        self.vi_n_iter = int(np.log(1. / (epsilon * (1. - self.gamma))) / (1. - self.gamma))  # Nb of value iterations
        self.delta_r = delta_r

        self.prev_s = None
        self.prev_a = None
        self.U, self.R, self.T, self.counter = self.empty_memory_structure()

        # Lifelong Learning memories
        self.U_memory = []
        self.R_memory = []
        self.T_memory = []

        self.U_lip = []
        self.update_lipschitz_upper_bounds()

    def reset(self):
        """
        Reset the attributes to initial state (called between instances).
        Save the previous model.
        :return: None
        """
        if len(self.counter) > 0:  # Save previously learned model
            self.update_memory()

        self.prev_s = None
        self.prev_a = None
        self.U, self.R, self.T, self.counter = self.empty_memory_structure()

        self.update_lipschitz_upper_bounds()

    def end_of_episode(self):
        """
        Reset between episodes within the same MDP.
        :return: None
        """
        self.prev_s = None
        self.prev_a = None

    def empty_memory_structure(self):
        """
        Empty memory structure:
        U[s][a] (float): upper-bound on the Q-value
        R[s][a] (list): list of collected rewards
        T[s][a][s'] (int): number of times the transition has been observed
        counter[s][a] (int): number of times the state action pair has been sampled
        :return: R, T, counter
        """
        return defaultdict(lambda: defaultdict(lambda: self.r_max / (1. - self.gamma))), \
               defaultdict(lambda: defaultdict(list)), \
               defaultdict(lambda: defaultdict(lambda: defaultdict(int))), \
               defaultdict(lambda: defaultdict(int))

    def display(self):
        """
        Display info about the attributes.
        """
        print('Displaying R-MAX agent :')
        print('Action space           :', self.actions)
        print('Number of actions      :', len(self.actions))
        print('Gamma                  :', self.gamma)
        print('Count threshold        :', self.count_threshold)

    def is_known(self, s, a):
        return self.counter[s][a] >= self.count_threshold

    def get_nb_known_sa(self):
        return sum([self.is_known(s, a) for s, a in self.counter.keys()])

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
        u_min = defaultdict(lambda: defaultdict(lambda: self.r_max / (1. - self.gamma)))
        for a in self.actions:
            u_min[s][a] = self.U[s][a]
            if len(self.U_lip) > 0:
                for u in self.U_lip:
                    if u[s][a] < u_min[s][a]:
                        u_min[s][a] = u[s][a]
        return u_min

    def update(self, s, a, r, s_p):
        """
        Updates transition and reward dictionaries with the input transition
        tuple if the corresponding state-action pair is not known enough.
        If a new state-action pair is known, update both the R-Max upper-bound
        and the Lipschitz upper-bounds.
        :param s: state
        :param a: action
        :param r: reward
        :param s_p: next state
        :return: None
        """
        if s is not None and a is not None:
            if self.counter[s][a] < self.count_threshold:
                self.counter[s][a] += 1
                self.R[s][a] += [r]
                self.T[s][a][s_p] += 1
                if self.counter[s][a] == self.count_threshold:
                    self.update_rmax_upper_bound()
                    self.update_lipschitz_upper_bounds()

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

    def greedy_action(self, s, upper_bound):
        """
        Compute the greedy action wrt the input upper bound.
        :param s: state at which the upper-bound is evaluated
        :param upper_bound: input upper-bound
        :return: return the greedy action.
        """
        a_star = random.choice(self.actions)
        u_star = upper_bound[s][a_star]
        for a in self.actions:
            u_s_a = upper_bound[s][a]
            if u_s_a > u_star:
                u_star = u_s_a
                a_star = a
        return a_star

    def update_rmax_upper_bound(self):
        """
        Update the upper bound on the Q-value function.
        Called when a new state-action pair is known.
        :return: None
        """
        for i in range(self.vi_n_iter):
            for s in self.R:
                for a in self.R[s]:
                    n_s_a = float(self.counter[s][a])
                    r_s_a = sum(self.R[s][a]) / n_s_a

                    weighted_next_upper_bound = 0.
                    for s_p in self.T[s][a]:
                        a_p = self.greedy_action(s_p, self.U)
                        weighted_next_upper_bound += self.U[s_p][a_p] * self.T[s][a][s_p] / n_s_a

                    self.U[s][a] = r_s_a + self.gamma * weighted_next_upper_bound

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
        ''' Note: different from LRMax '''
        # 1. Separate state-action pairs
        s_a_kk, s_a_ku, s_a_uk = self.separate_state_action_pairs(R_mem)

        # 2. Compute models distances upper-bounds
        d_dict = self.models_distances(R_mem, s_a_kk, s_a_ku, s_a_uk)

        # 3. Compute the Q-values gap with dynamic programming
        gap = self.q_values_gap(d_dict, T_mem, s_a_kk, s_a_ku, s_a_uk)

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

    def models_distances(self, R_mem, s_a_kk, s_a_ku, s_a_uk):
        ''' Note: different from LRMax '''
        # Initialize model's distances upper-bounds
        d_dict = defaultdict(lambda: defaultdict(lambda: self.delta_r))

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
            d_dict[s][a] = min(self.delta_r, max(self.r_max - r_s_a, r_s_a))
            assert self.count_threshold == self.counter[s][a]  # TODO remove after testing

        # Compute model's distances upper-bounds for unknown-known (s, a)
        for s, a in s_a_uk:
            r_s_a_mem = sum(R_mem[s][a]) / float(self.count_threshold)
            d_dict[s][a] = min(self.delta_r, max(self.r_max - r_s_a_mem, r_s_a_mem))
            assert self.count_threshold == len(R_mem[s][a])  # TODO remove after testing

        return d_dict

    def q_values_gap(self, d_dict, T_mem, s_a_kk, s_a_ku, s_a_uk):
        ''' Note: different from LRMax '''
        gap_max = self.delta_r / (1. - self.gamma)
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
        U = defaultdict(lambda: defaultdict(lambda: (self.delta_r + self.r_max) / (1. - self.gamma)))
        for s in gap:
            for a in gap[s]:
                U[s][a] = U_mem[s][a] + gap[s][a]
        return U
