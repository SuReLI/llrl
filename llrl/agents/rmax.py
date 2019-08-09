"""
Implementation of an R-Max agent [Brafman and Tennenholtz 2003]
Use Value Iteration to compute the R-Max upper-bound following [Strehl et al 2009].

Changes compared to Dave's original RMax:

1) Use of Value Iteration for upper-bound computation with provable precision + faster computation;

2) Only use learned transition model when state-action pair is known;

3) Directly store transition probabilities and averaged reward signal instead of counters and list of sampled rewards
   resulting in faster computation and easier interpretability;

4) Single visit counter (lower memory requirements);

5) count_threshold can be replaced by (epsilon, delta) for epsilon accurate models in L1 norm
   with probability at least 1 - delta.
"""

import random
import numpy as np
from collections import defaultdict

from simple_rl.agents.AgentClass import Agent


def compute_n_model_samples_high_confidence(epsilon, delta, n_states):
    """
    Compute the number of model samples required for epsilon accurate model in L1-norm
    with probability at least 1 - delta.
    :param epsilon: (float)
    :param delta: (float)
    :param n_states: (int)
    :return: (int)
    """
    m_r_min = np.log(2. / delta) / (2. * epsilon ** 2.)
    m_t_min = 2. * (np.log(2. ** float(n_states) - 2.) - np.log(delta)) / (epsilon ** 2.)
    return int(max(m_r_min, m_t_min)) + 1


class RMax(Agent):
    """
    Implementation of an R-Max agent [Brafman and Tennenholtz 2003]
    Use Value Iteration to compute the R-Max upper-bound following [Strehl et al 2009].
    """

    def __init__(
            self,
            actions,
            gamma=0.9,
            count_threshold=None,
            epsilon_q=0.1,
            epsilon_m=None,
            delta=None,
            n_states=None,
            name="RMax"
    ):
        """
        :param actions: action space of the environment
        :param gamma: (float) discount factor
        :param count_threshold: (int) count after which a state-action pair is considered known
        (only set count_threshold if delta and epsilon are not defined)
        :param epsilon_q: (float) precision of value iteration algorithm for Q-value computation
        :param epsilon_m: (float) precision of the learned models in L1 norm
        :param delta: (float) models are learned epsilon_m-closely with probability at least 1 - delta
        :param n_states: (int) number of states
        :param name: (str)
        """
        if count_threshold is None:
            if delta is None or epsilon_m is None or n_states is None:
                raise ValueError('Please set either `count_threshold` or `delta`, `epsilon_m` and `n_states`.')
        else:
            if delta is not None or epsilon_m is not None or n_states is not None:
                raise ValueError('Please set either `count_threshold` or `delta`, `epsilon_m` and `n_states`.')

        Agent.__init__(self, name=name, actions=actions, gamma=gamma)
        self.r_max = 1.0

        if count_threshold is None:
            self.count_threshold = compute_n_model_samples_high_confidence(epsilon_m, delta, n_states)
        else:
            self.count_threshold = count_threshold

        self.vi_n_iter = int(np.log(1. / (epsilon_q * (1. - self.gamma))) / (1. - self.gamma))  # Nb of value iterations

        self.U, self.R, self.T, self.counter = self.empty_memory_structure()
        self.prev_s = None
        self.prev_a = None

    def reset(self):
        """
        Reset the attributes to initial state (called between instances).
        :return: None
        """
        self.U, self.R, self.T, self.counter = self.empty_memory_structure()
        self.prev_s = None
        self.prev_a = None

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
        for _ in range(self.vi_n_iter):
            for s in self.R:
                for a in self.R[s]:
                    if self.is_known(s, a):
                        u_p = 0.
                        for s_p in self.T[s][a]:
                            u_p += max(self.U[s_p][a_p] for a_p in self.actions) * self.T[s][a][s_p]

                        self.U[s][a] = self.R[s][a] + self.gamma * u_p
