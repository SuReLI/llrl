import random
import numpy as np
from collections import defaultdict

from simple_rl.agents.AgentClass import Agent


class LRMax(Agent):
    """
    Lipschitz R-Max agent
    """

    def __init__(self, actions, gamma=0.9, count_threshold=1, epsilon=0.1, name="LRMax-e"):
        name = name + str(epsilon) if name[-2:] == "-e" else name
        Agent.__init__(self, name=name, actions=actions, gamma=gamma)
        self.nA = len(self.actions)
        self.r_max = 1.0
        self.count_threshold = count_threshold
        self.vi_n_iter = int(np.log(1. / (epsilon * (1. - self.gamma))) / (1. - self.gamma))  # Nb of value iterations

        self.prev_s = None
        self.prev_a = None
        self.U, self.R, self.T, self.counter = self.empty_memory_structure()

        self.U_lip = self.init_lipschitz_upper_bounds()

        # Lifelong Learning memories
        self.U_memory = []
        self.R_memory = []
        self.T_memory = []


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

        self.U_lip = self.init_lipschitz_upper_bounds()

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

    def init_lipschitz_upper_bounds(self):
        """
        Initialize the Lipschitz upper-bound for all the previous tasks.
        :return: None
        """
        n_prev_mdps = len(self.U_memory)
        if n_prev_mdps == 0:
            return []
        else:
            for i in range(n_prev_mdps):
                u_i =

    def compute_lipschitz_upper_bound(self, U_mem, R_mem, T_mem):
        # Initialize model's distances upper-bounds
        d_dict = defaultdict(lambda: defaultdict(lambda: self.r_max * (1. + self.gamma) / (1. - self.gamma)))

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
                if not self.is_known(s, a): # (s, a) only known in previous MDP
                    s_a_uk.append((s, a))

        # Compute model's distances upper-bounds
        for s, a in s_a_kk:
            r_s_a = sum(self.R[s][a]) / float(len(self.R[s][a]))
            r_s_a_mem = sum(R_mem[s][a]) / float(len(R_mem[s][a]))

            weighted_sum = 0.
            for s_p in self.T[s][a]:
                if s_p in T_mem[s][a]:
                    weighted_sum +=  # TODO

            d_dict[s][a] = abs(r_s_a - r_s_a_mem) + self.gamma * weighted_sum
        for s, a in s_a_ku:
            # TODO
        for s, a in s_a_uk:
            # TODO

    def display(self):
        """
        Display info about the attributes.
        """
        print('Displaying R-MAX agent :')
        print('Action space           :', self.actions)
        print('Number of actions      :', self.nA)
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

        # local_upper_bound = self.compute_min_upper_bound(s)  # TODO need?
        a = self.greedy_action(s, self.U)

        self.prev_a = a
        self.prev_s = s

        return a

    def update(self, s, a, r, s_p):
        """
        Updates transition and reward dictionaries with the input transition
        tuple if the corresponding state-action pair is not known enough.
        :param s: int state
        :param a: int action
        :param r: float reward
        :param s_p: int next state
        :return: None
        """
        if s is not None and a is not None:
            if self.counter[s][a] < self.count_threshold:
                self.counter[s][a] += 1
                self.R[s][a] += [r]
                self.T[s][a][s_p] += 1
                if self.counter[s][a] == self.count_threshold:
                    self.update_rmax_upper_bound()

    def update_memory(self):
        """
        Update the memory:
        Store the rewards, transitions and upper-bounds for the known state-action pairs
        respectively in R_memory, T_memory and U_memory.
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

    def compute_min_upper_bound(self, s):

        #TODO
        return 0  # TODO remove

    def greedy_action(self, s, upper_bound):
        """
        Compute the greedy action wrt the input upper bound.
        :param s: state
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

                    s_p_dict = self.T[s][a]
                    weighted_next_upper_bound = 0.
                    for s_p in s_p_dict:
                        a_p = self.greedy_action(s_p, self.U)
                        weighted_next_upper_bound += self.U[s_p][a_p] * s_p_dict[s_p] / n_s_a

                    self.U[s][a] = r_s_a + self.gamma * weighted_next_upper_bound
