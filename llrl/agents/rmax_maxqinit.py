"""
Implementation of an MaxQInit [Abel et al 2018]
"""

import copy
import random
import numpy as np
from collections import defaultdict

from simple_rl.agents.AgentClass import Agent


class RMaxMaxQInit(Agent):
    def __init__(
            self,
            actions,
            gamma=0.9,
            count_threshold=1,
            epsilon=0.1,
            min_sampling_probability=0.1,
            delta=0.05,
            name="RMax-MaxQInit"
    ):
        """
        :param actions: action space of the environment
        :param gamma: (float) discount factor
        :param count_threshold: (int) count after which a state-action pair is considered known
        :param epsilon: (float) precision of value iteration algorithm
        :param min_sampling_probability: (float) minimum sampling probability of an environment
        :param delta: (float) uncertainty degree on the maximum model's distance of a state-action pair
        :param name: (str)
        """
        Agent.__init__(self, name=name, actions=actions, gamma=gamma)
        self.r_max = 1.0
        self.count_threshold = count_threshold
        self.vi_n_iter = int(np.log(1. / (epsilon * (1. - self.gamma))) / (1. - self.gamma))  # Nb of value iterations

        self.U, self.R, self.T, self.counter = self.empty_memory_structure()
        self.prev_s = None
        self.prev_a = None

        self.SA_memory = []
        self.U_memory = []  # Upper-bounds on the Q-values of previous MDPs
        self.n_required_mdps = np.log(delta) / np.log(1. - min_sampling_probability)

    def reset(self):
        """
        Reset the attributes to initial state (called between instances).
        :return: None
        """
        for s in self.R:
            for a in self.R[s]:
                if self.is_known(s, a):
                    self.SA_memory.append((s, a))
        self.U_memory.append(copy.deepcopy(self.U))

        self.U, self.R, self.T, self.counter = self.empty_memory_structure()
        self.prev_s = None
        self.prev_a = None

        if len(self.U_memory) > self.n_required_mdps:
            self.update_max_q_init_upper_bound()

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
        R[s][a] (float): average reward
        T[s][a][s'] (float): probability of the transition
        counter[s][a] (int): number of times the state action pair has been sampled
        :return: U, R, T, counter
        """
        return defaultdict(lambda: defaultdict(lambda: self.r_max / (1.0 - self.gamma))), \
               defaultdict(lambda: defaultdict(float)), \
               defaultdict(lambda: defaultdict(lambda: defaultdict(float))), \
               defaultdict(lambda: defaultdict(int))

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
                normalizer = 1. / float(self.counter[s][a])

                self.R[s][a] = self.R[s][a] + normalizer * (r - self.R[s][a])
                self.T[s][a][s_p] = self.T[s][a][s_p] + normalizer * (1. - self.T[s][a][s_p])
                for _s_p in self.T[s][a]:
                    if _s_p not in [s_p]:
                        self.T[s][a][_s_p] = self.T[s][a][_s_p] * (1 - normalizer)

                if self.counter[s][a] == self.count_threshold:
                    self.update_rmax_upper_bound()

    def greedy_action(self, s, f):
        """
        Compute the greedy action wrt the input function of (s, a).
        :param s: state at which the upper-bound is evaluated
        :param f: input function of (s, a)
        :return: return the greedy action.
        """
        a_star = random.choice(self.actions)
        u_star = f[s][a_star]
        for a in self.actions:
            u_s_a = f[s][a]
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
                    if self.is_known(s, a):
                        r_s_a = self.R[s][a]

                        weighted_next_upper_bound = 0.
                        for s_p in self.T[s][a]:
                            # TODO use max
                            weighted_next_upper_bound += self.U[s_p][self.greedy_action(s_p, self.U)] * self.T[s][a][s_p]

                        self.U[s][a] = r_s_a + self.gamma * weighted_next_upper_bound

    def update_max_q_init_upper_bound(self):
        """
        Update the bound on the Q-value with the MaxQInit method.
        :return: None
        """
        for s, a in self.SA_memory:
            self.U[s][a] = max([u[s][a] for u in self.U_memory])
