import numpy as np

from simple_rl.mdp.MDPClass import MDP
from simple_rl.mdp.StateClass import State


class TwoStates(MDP):
    """
    Implementation of a two-states MDP for simple experiments.
    """

    def __init__(self, rewards=(0., 1.), proba=(.5, 0.)):
        self.nS = 2
        self.nA = 2
        self.rewards = rewards
        self.proba = proba
        self.states = range(self.nS)
        self.actions = range(self.nA)
        self.T = self.generate_transition_matrix()
        MDP.__init__(self, self.actions, self._transition_func, self._reward_func, init_state=State(0))

    def __str__(self):
        return "two-states"

    def _transition_func(self, s, a):
        """
        Transition function sampling a resulting state from the application of the input action at the input state.

        :param s: input state
        :param a: input action
        :return: sampled next state
        """
        return State(np.random.choice(list(self.states), p=self.T[s.data, a]))

    def _reward_func(self, s, a):
        """
        Reward function sampling a reward signal from the application of the input action at the input state.

        :param s: input state
        :param a: input action
        :return: sampled reward
        """
        return self.rewards[s.data]

    def generate_transition_matrix(self):
        T = np.zeros(shape=(self.nS, self.nA, self.nS), dtype=float)
        for s in self.states:
            for a in self.actions:
                T[s, a] = [self.proba[a], 1. - self.proba[a]]
        return T
