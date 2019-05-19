import numpy as np
from collections import defaultdict

from simple_rl.mdp.MDPClass import MDP
from simple_rl.mdp.StateClass import State


class NStates(MDP):
    """
    Implementation of a n-states MDP.
    """

    def __init__(self, ns, r):
        self.nS = ns
        self.nA = 2
        self.states = range(self.nS)
        self.actions = range(self.nA)
        self.T = self.generate_transition_matrix()
        self.R = r
        MDP.__init__(self, self.actions, self._transition_func, init_state=0)

    def __str__(self):
        return "n-states"

    def _transition_func(self, s, a):
        """
        Transition function sampling a resulting state from the application of the input action at the input state and
        the collected reward along the transition.

        :param s: input state
        :param a: input action
        :return: sampled next state, reward
        """
        s_p_dist = self.transition_probability_distribution(s, a)
        s_p = np.random.choice(self.states, p=s_p_dist)
        r = self.expected_reward(s, a)
        return s_p, r

    def generate_transition_matrix(self):
        tm = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        for s in self.states:
            for a in self.actions:
                for s_p in self.states:
                    if a == 0:  # Noop
                        tm[s][a][s_p] = 1. if s_p == s else 0.
                    else:
                        tm[s][a][s_p] = 1. if (s_p == s + 1 or (s == self.nS - 1 and s_p == 0)) else 0.
        return tm

    def transition_probability_distribution(self, s, a):
        return np.array([self.T[s][a][s_p] for s_p in self.states])

    def expected_reward(self, s, a):
        return self.R[s][a]

    def number_of_states(self):
        return self.nS

    def number_of_actions(self):
        return self.nA
